import { toast as sonner } from 'sonner';
import type { AddImagesResult } from '../types';
import type { YoloImportResult } from './useStore';

/**
 * 全局提示：底层用 sonner（shadcn 推荐方案），保留轻量 API 方便调用。
 * 持续时间与堆叠交给 sonner 默认行为，视觉随主题变量走。
 */
export const toast = {
  info: (text: string) => sonner.info(text),
  /**
   * 传入 id 时 sonner 会**复用同一条**提示而不是新堆一条。
   * 连续动作（如点击标注一下出一个框）必须带 id，否则每点一下堆一条、瞬间刷屏。
   */
  success: (text: string, id?: string) => sonner.success(text, id ? { id } : undefined),
  warning: (text: string) => sonner.warning(text, { duration: 6000 }),
  error: (text: string) => sonner.error(text, { duration: 4200 }),
};

/** 导入图片的统一反馈：区分「没有可用图片」「都已导入过」「把丢失的图接了回去」 */
export function reportAddImages(result: AddImagesResult): void {
  const { added, skipped, revived } = result;
  if (added === 0 && revived === 0) {
    toast.error(skipped > 0 ? '这些图片已经导入过了' : '没有找到可用图片');
    return;
  }

  const parts: string[] = [];
  if (revived > 0) parts.push(`接回 ${revived} 张（原标注已保留）`);
  if (added > 0) parts.push(`新增 ${added} 张`);
  if (skipped > 0) parts.push(`跳过 ${skipped} 张重复`);
  toast.success(parts.join('，'));
}

/**
 * 「导入已标注 YOLO 数据集」的结果描述。
 * 落地页与编辑器各有一条导入入口，文案必须一致，否则用户会以为其中一条坏了。
 * 每一项都要如实上报，尤其是「有歧义而故意跳过」的情况 ——
 * 静默少导入比报错更难排查。
 */
export function describeImport(res: YoloImportResult): string {
  const parts: string[] = [];
  if (res.added > 0) parts.push(`导入 ${res.added} 张图`);
  if (res.skipped > 0) parts.push(`跳过 ${res.skipped} 张重复`);
  if (res.newLabels > 0) parts.push(`新建 ${res.newLabels} 个类别`);
  if (res.boxes > 0) parts.push(`读入 ${res.boxes} 个已有标注`);
  if (res.unresolved > 0) parts.push(`${res.unresolved} 张尺寸读取失败、未套用标注`);
  if (res.ambiguous > 0) {
    parts.push(`${res.ambiguous} 张因标注文件重名未套用标注（同名会套错图，已主动跳过）`);
  }
  return parts.join('，');
}
