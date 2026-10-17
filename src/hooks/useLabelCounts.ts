import { useMemo } from 'react';
import { useStore } from '../store/useStore';

/**
 * 每个类别各有多少个框。
 *
 * 底栏标签条、右侧类别列表、框旁的标签菜单都要这份数据，
 * 之前三处各自把「所有图片的所有标注」遍历一遍 —— 图集一大会变成三倍的 O(图×框)。
 *
 * 这里做两级缓存：
 *  1. 弱引用缓存：按 images 数组引用，store 里 images 都是不可变更新，引用变则内容必变。
 *  2. 版本号：labelRevision 只在「标签集合或归属真的变了」时才 +1。
 *
 * 为什么需要第 2 级：单纯按引用缓存的话，**拖动一个框的边**这种纯几何操作
 * 也会让 images 换引用 → 缓存必 miss → 6 万次迭代（3000 张 × 20 框）全白跑，
 * 而类别计数其实一个数都没变。密集标注时这是每帧都在付的冤枉钱。
 */
/**
 * 缓存**按版本号**存，而不是按 images 数组引用存。
 *
 * 旧实现用 `WeakMap<ImageItem[], ...>`：key 是 images 数组引用，
 * 而 store 全是不可变更新 —— **拖一下框边也会换掉数组引用**，
 * 于是恰好在最需要缓存的场景（密集标注时每帧都在拖动）必然 miss，
 * 6 万次迭代全白跑，注释里写的第二级缓存等于没写。
 *
 * 现在：只在 labelRevision 变化时重算，其余情况直接返回同一个 Map 对象
 * （引用不变 → 消费方也不会因它而重渲染）。
 */
let cachedRev = -1;
let cachedMap: Map<string, number> = new Map();

export function useLabelCounts(): Map<string, number> {
  // 只订阅版本号这个**数字**，不订阅 images。
  // 订阅 images 会让调用方（底栏 / 右栏 / 框旁标签菜单）在每次标注改动时都重渲染，
  // 哪怕类别计数一个数都没变。真要重算时再用 getState() 现读一次。
  const labelRevision = useStore((s) => s.labelRevision);
  return useMemo(() => {
    if (cachedRev === labelRevision) return cachedMap;

    const map = new Map<string, number>();
    for (const img of useStore.getState().images) {
      for (const a of img.annotations) {
        // 未分类（labelId 为空串）不计入任何类别
        if (!a.labelId) continue;
        map.set(a.labelId, (map.get(a.labelId) ?? 0) + 1);
      }
    }
    cachedRev = labelRevision;
    cachedMap = map;
    return map;
  }, [labelRevision]);
}
