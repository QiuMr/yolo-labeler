import { clsx, type ClassValue } from 'clsx';
import { extendTailwindMerge } from 'tailwind-merge';

/**
 * 本项目在 index.css 的 @theme 里自定义了 5 级字号（--text-micro … --text-display），
 * Tailwind 据此生成 text-micro / text-caption / text-body / text-title / text-display。
 *
 * ⚠️ 坑：tailwind-merge 不认识这些名字，解析 text-* 时先试已知字号、再试颜色，
 * 两边都落空就**归进「文字颜色」兜底组**。后果是它们与被判成同组的真颜色类互删，
 * 而且不报任何错 ——
 *
 *   cn('text-body text-primary-foreground')  → 'text-primary-foreground'   // 字号没了
 *   cn('text-primary-foreground text-body')  → 'text-body'                 // 白字没了
 *
 * 真实症状：顶栏「导出」按钮的 base 里 text-body 排在 variant 的
 * text-primary-foreground 之后，把白字类挤掉了。文字回落到 --foreground 深藏青，
 * 图标（currentColor）跟着一起变深，于是蓝底上出现一个「蓝黑」按钮。
 *
 * 修法：用 extendTailwindMerge 把这 5 个类**显式注册进 font-size 组**，
 * 让它们和真字号（text-sm 等）同组、与颜色组彻底分开。
 * 这是唯一的根因层修法 —— 全站任何 cn() 调用点都自愈，
 * 且调用方不必关心类名顺序，后续新增自定义字号也只需在这里补一行。
 */
const FONT_SIZE_CLASSES = [
  'text-micro',
  'text-caption',
  'text-body',
  'text-title',
  'text-display',
];

const twMerge = extendTailwindMerge({
  extend: {
    classGroups: {
      'font-size': FONT_SIZE_CLASSES,
    },
  },
});

export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}
