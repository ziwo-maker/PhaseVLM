export function transCoord(point) {
  return [point[0], -point[1]];
}

export function statusToColor(status) {
  const LIGHT_RED = 0xdb635e;
  const LIGHT_GREEN = 0x85ee00;
  switch (status) {
    case 'r':
      return LIGHT_RED;
    case 'g':
      return LIGHT_GREEN;
    default:
      return 0x808080;
  }
}

export function stringHash(str) {
  let hash = 0;
  let p = 127, p_pow = 1;
  let m = 1e9 + 9;
  for (let i = 0; i < str.length; i++) {
    hash = (hash + str.charCodeAt(i) * p_pow) % m;
    p_pow = (p_pow * p) % m;
  }
  return hash;
}

export function loadFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => resolve(e.target.result);
    reader.onerror = (e) => reject(e);
    reader.readAsText(file);
  });
}

// 计算点到线段的最短距离
export function pointToSegmentDistance(point, segStart, segEnd) {
  const px = point.x, py = point.y;
  const x1 = segStart.x, y1 = segStart.y;
  const x2 = segEnd.x, y2 = segEnd.y;
  
  const dx = x2 - x1;
  const dy = y2 - y1;
  
  // 如果线段是一个点
  if (dx === 0 && dy === 0) {
    return Math.sqrt((px - x1) ** 2 + (py - y1) ** 2);
  }
  
  // 计算投影参数t
  const t = Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)));
  
  // 计算线段上最近的点
  const closestX = x1 + t * dx;
  const closestY = y1 + t * dy;
  
  return Math.sqrt((px - closestX) ** 2 + (py - closestY) ** 2);
}

// 根据停留车辆数量计算道路颜色（浅蓝色->黄色->橙色->红色渐变）
export function getTrafficColor(stoppedCarCount) {
  if (stoppedCarCount === 0) return 0xADD8E6;  // 浅蓝色
  
  if (stoppedCarCount <= 2) {
    // 0-2辆：浅蓝色到浅黄色
    const t = stoppedCarCount / 2;
    return interpolateColor(0xADD8E6, 0xFFFFAA, t);
  } else if (stoppedCarCount <= 5) {
    // 3-5辆：浅黄色到黄色
    const t = (stoppedCarCount - 2) / 3;
    return interpolateColor(0xFFFFAA, 0xFFFF00, t);
  } else if (stoppedCarCount <= 8) {
    // 6-8辆：黄色到橙色
    const t = (stoppedCarCount - 5) / 3;
    return interpolateColor(0xFFFF00, 0xFF8800, t);
  } else {
    // 8辆以上：橙色到红色
    const t = Math.min((stoppedCarCount - 8) / 7, 1);
    return interpolateColor(0xFF8800, 0xFF0000, t);
  }
}

// 颜色插值函数
export function interpolateColor(color1, color2, t) {
  const r1 = (color1 >> 16) & 0xff;
  const g1 = (color1 >> 8) & 0xff;
  const b1 = color1 & 0xff;
  
  const r2 = (color2 >> 16) & 0xff;
  const g2 = (color2 >> 8) & 0xff;
  const b2 = color2 & 0xff;
  
  const r = Math.round(r1 + (r2 - r1) * t);
  const g = Math.round(g1 + (g2 - g1) * t);
  const b = Math.round(b1 + (b2 - b1) * t);
  
  return (r << 16) | (g << 8) | b;
}
