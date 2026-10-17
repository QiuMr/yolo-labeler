let counter = 0;

/** 生成局部唯一 id（比 Math.random 更可控，避免极端情况下碰撞） */
export function uid(prefix = ''): string {
  counter = (counter + 1) % 1_000_000;
  const rand = Math.random().toString(36).slice(2, 8);
  return `${prefix}${Date.now().toString(36)}${counter.toString(36)}${rand}`;
}
