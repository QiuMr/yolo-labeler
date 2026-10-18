import type { Annotation, IRect, Label } from '../types';
import { clampRectToImage } from './geometry';

/**
 * AI 落框的纯逻辑（与 store 解耦，便于直接单测）。
 *
 * 抽出来的原因是这里有一条**会影响最终数据集正确性**的规则：重复框治理。
 * 「调低阈值再跑一遍」是标注调优的标准动作，而它必然会在同一张图上产出与已有框
 * 几乎重合的新框。原来无论跑多少次都是纯追加，于是每个物体挂两个框、用户只能手工删。
 * 这类规则躲在 store 的闭包里没法测，必须搬到能被断言的地方。
 */

/**
 * 认作「同一个物体」的 IoU 阈值。
 * 取 0.6 而不是 0.5：同类物体真实重叠（并排的两辆车、货架上紧挨的同类商品）
 * 常见到 0.4~0.5，压得太低会开始误删真正的不同物体 —— 漏删只是多一个框要手工点掉，
 * 误删却会静默丢掉一个真值，两者代价不对等，所以宁可偏向保守。
 */
export const AI_DEDUP_IOU = 0.6;

/** 两个矩形的高度重叠度（交并比）。任一维不相交即 0 */
export function iouOf(a: IRect, b: IRect): number {
  const x0 = Math.max(a.x, b.x);
  const y0 = Math.max(a.y, b.y);
  const x1 = Math.min(a.x + a.width, b.x + b.width);
  const y1 = Math.min(a.y + a.height, b.y + b.height);
  const iw = x1 - x0;
  const ih = y1 - y0;
  if (iw <= 0 || ih <= 0) return 0;
  const inter = iw * ih;
  const union = a.width * a.height + b.width * b.height - inter;
  return union > 0 ? inter / union : 0;
}

/** 一条 AI 结果（来自 detector 或点击标注） */
export interface AiBoxEntry {
  rect: IRect;
  className: string;
  /** 模型置信度 0~1；拿不到就留空，不编造 */
  score?: number;
}

export interface BuildAiBoxesOptions {
  /** 这张图**已有的**框（用于去重） */
  existing: readonly Annotation[];
  entries: readonly AiBoxEntry[];
  /**
   * 整批共享的标签表，**会被就地追加**新建的类别。
   * 批量标注必须让所有图共用同一份：每张图各拿一份副本的话，
   * 同一个类名会被建出多个 id 不同的标签，左侧标签栏立刻裂成好几行。
   */
  labels: Label[];
  /** 类名 → 标签 id，与 labels 同步维护 */
  nameToId: Map<string, string>;
  width: number;
  height: number;
  /** 新建类别（由调用方注入，store 里是 makeLabel） */
  createLabel: (name: string, index: number) => Label;
  createId: (prefix: string) => string;
}

/**
 * 把一批 AI 结果合并成可插入的标注框。
 * 返回新增的框与**被判定为重复而跳过**的条数 —— 跳过必须能被上报，
 * 否则用户会觉得「跑了 12 个目标怎么只出来 9 个」。
 */
export function buildAiAnnotations(o: BuildAiBoxesOptions): {
  anns: Annotation[];
  skipped: number;
} {
  const { existing, entries, labels, nameToId, width, height, createLabel, createId } = o;
  const anns: Annotation[] = [];
  let skipped = 0;

  for (const en of entries) {
    let labelId = nameToId.get(en.className);
    if (!labelId) {
      const label = createLabel(en.className, labels.length);
      labels.push(label);
      nameToId.set(en.className, label.id);
      labelId = label.id;
    }

    const rect = clampRectToImage(en.rect, width, height);
    // 和「手工画框」同一条底线：退化的框不入库
    if (rect.width < 2 || rect.height < 2) continue;

    // 只和**同类**框比：不同类别高度重叠是正常情况（人骑在自行车上、杯子放在桌上）
    const dup =
      existing.some((a) => a.labelId === labelId && iouOf(a.rect, rect) >= AI_DEDUP_IOU) ||
      anns.some((a) => a.labelId === labelId && iouOf(a.rect, rect) >= AI_DEDUP_IOU);
    if (dup) {
      skipped += 1;
      continue;
    }

    anns.push({ id: createId('an_'), labelId, rect, score: en.score, source: 'ai' });
  }

  return { anns, skipped };
}
