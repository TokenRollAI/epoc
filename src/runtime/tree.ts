import type { ScopeNode } from "../types.ts";
import { nextId, nowIso } from "../util/misc.ts";
import type { BlockStore } from "./blocks.ts";
import type { Journal } from "./stores.ts";

/**
 * Spine-inspired persistent subtask tree (`ScopeTree`).
 *
 * Aligns context management with nested subtask boundaries and provider prefix
 * caching:
 *   - While a child scope is `active`, turns append monotonically -> 100% prefix
 *     cache hits inside the scope.
 *   - When the child scope completes (`commitSubmit`), all raw exploratory
 *     chains belonging to that scope (from the assistant turn that opened it
 *     through the tool result that closed it) are archived in one atomic wave,
 *     and a single immutable ScopeResult summary block is placed right after
 *     `entry_after_id` — popping the prompt back to the hot cached parent prefix!
 */
export class ScopeTree {
  private readonly nodes = new Map<string, ScopeNode>();
  private readonly stack: string[] = []; // active scope ids from root-most child -> deepest active
  private readonly blocks: BlockStore;
  private readonly journal: Journal;

  constructor(blocks: BlockStore, journal: Journal) {
    this.blocks = blocks;
    this.journal = journal;
  }

  activeScope(): ScopeNode | undefined {
    const id = this.stack[this.stack.length - 1];
    return id ? this.nodes.get(id) : undefined;
  }

  activeScopeId(): string | undefined {
    return this.stack[this.stack.length - 1];
  }

  activePath(): ScopeNode[] {
    return this.stack
      .map((id) => this.nodes.get(id))
      .filter((n): n is ScopeNode => n !== undefined);
  }

  get(id: string): ScopeNode | undefined {
    return this.nodes.get(id);
  }

  all(): ScopeNode[] {
    return Array.from(this.nodes.values());
  }

  /**
   * Opens a nested subtask scope. `openingBlockId` is the assistant message block
   * that emitted the `scope_open` tool call so that when the scope later closes,
   * the entire span starting at `openingBlockId` collapses cleanly back to the
   * block immediately before `openingBlockId`.
   */
  openScope(opts: {
    label: string;
    goal: string;
    openingBlockId?: string;
  }): ScopeNode {
    const id = nextId("scope");
    const parent = this.activeScope();

    let entryAfterId: string | undefined;
    if (opts.openingBlockId) {
      const idx = this.blocks.order.indexOf(opts.openingBlockId);
      entryAfterId = idx > 0 ? this.blocks.order[idx - 1] : undefined;
    } else if (this.blocks.order.length > 0) {
      entryAfterId = this.blocks.order[this.blocks.order.length - 1];
    }

    const node: ScopeNode = {
      id,
      parent_id: parent?.id,
      label: opts.label.trim() || id,
      goal: opts.goal.trim(),
      status: "active",
      entry_after_id: entryAfterId,
      block_ids: opts.openingBlockId ? [opts.openingBlockId] : [],
      child_scope_ids: [],
      created_at: nowIso(),
    };

    if (opts.openingBlockId) {
      const openBlock = this.blocks.get(opts.openingBlockId);
      if (openBlock && !openBlock.scope_id) {
        openBlock.scope_id = id;
      }
    }

    this.nodes.set(id, node);
    if (parent) parent.child_scope_ids.push(id);
    this.stack.push(id);

    this.journal.append("scope", {
      action: "open",
      scope_id: id,
      label: node.label,
    });
    return node;
  }

  /** Record a newly created block against the currently active scope, if any. */
  registerBlock(blockId: string): void {
    const active = this.activeScope();
    if (!active) return;
    if (!active.block_ids.includes(blockId)) {
      active.block_ids.push(blockId);
    }
  }

  /**
   * Collects the chain-closed set of visible non-protected blocks belonging to
   * `scopeId` (and any nested child summary blocks created inside it) so the
   * caller can collapse the scope via the transactional PatchEngine.
   */
  collectCollapsibleIds(scopeId: string): {
    node: ScopeNode;
    ids: string[];
  } | undefined {
    const node = this.nodes.get(scopeId);
    if (!node || node.status !== "active") return undefined;

    // Collect every visible block from right after `entry_after_id` to the end,
    // excluding system_prompt, user_message, task_state, and budget_report so
    // authoritative user messages arriving mid-scope are never swallowed.
    const startIdx = node.entry_after_id
      ? this.blocks.order.indexOf(node.entry_after_id) + 1
      : 0;

    const rawCandidates: string[] = [];
    for (let i = Math.max(0, startIdx); i < this.blocks.order.length; i++) {
      const b = this.blocks.get(this.blocks.order[i]);
      if (!b || b.visibility !== "model") continue;
      if (
        b.kind === "system_prompt" ||
        b.kind === "user_message" ||
        b.kind === "task_state" ||
        b.kind === "budget_report"
      ) {
        continue;
      }
      rawCandidates.push(b.id);
    }

    const closedIds = this.blocks.expandToChainClosure(rawCandidates).filter((id) => {
      const b = this.blocks.get(id);
      return (
        b !== undefined &&
        b.visibility === "model" &&
        b.kind !== "system_prompt" &&
        b.kind !== "user_message" &&
        b.kind !== "task_state" &&
        b.kind !== "budget_report"
      );
    });

    return { node, ids: closedIds };
  }

  markCompleted(
    scopeId: string,
    summary: string,
    keyFacts: string[],
    summaryBlockId?: string,
    freedTokens = 0,
  ): void {
    const node = this.nodes.get(scopeId);
    if (!node) return;
    node.status = "completed";
    node.submitted_result = summary;
    node.key_facts = keyFacts;
    node.summary_block_id = summaryBlockId;
    const idx = this.stack.lastIndexOf(scopeId);
    if (idx !== -1) {
      // Pop this scope and any nested unclosed children above it.
      for (let i = this.stack.length - 1; i >= idx; i--) {
        const poppedId = this.stack[i];
        const popped = this.nodes.get(poppedId);
        if (popped && popped.status === "active") {
          popped.status = poppedId === scopeId ? "completed" : "aborted";
        }
      }
      this.stack.splice(idx);
    }
    this.journal.append("scope", {
      action: "submit",
      scope_id: scopeId,
      label: node.label,
      freed_tokens: freedTokens,
    });
  }
}
