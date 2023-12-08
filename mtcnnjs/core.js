
const minFaceSize = 20; // Adjust based on the smallest face size you want to detect

function nonMaximumSuppression(boxes, threshold, method = 'Union', logKey) {
  if (boxes.length === 0) {
    return [];
  }

  // Calculate the area of each box
  const areas = boxes.map(box => (box.x2 - box.x1) * (box.y2 - box.y1));

  let sortedIndices = boxes.map((_, index) => index)
    .sort((a, b) => boxes[a].score - boxes[b].score);

  const pick = [];
  console.info(`${logKey} Before:`, sortedIndices.map(idx => boxes[idx]))

  while (sortedIndices.length > 0) {
    const currentIdx = sortedIndices.pop(); // Compare last box in the array (highest score) to the rest
    const currentBox = boxes[currentIdx];
    pick.push(currentBox);

    sortedIndices = sortedIndices.filter(idx => {
      const box = boxes[idx];

      const xx1 = Math.max(currentBox.x1, box.x1);
      const yy1 = Math.max(currentBox.y1, box.y1);
      const xx2 = Math.min(currentBox.x2, box.x2);
      const yy2 = Math.min(currentBox.y2, box.y2);

      const w = Math.max(0, xx2 - xx1);
      const h = Math.max(0, yy2 - yy1);
      const inter = w * h;

      const o = (method === 'Min')
        ? inter / Math.min(areas[currentIdx], areas[idx])
        : inter / (areas[currentIdx] + areas[idx] - inter);

      // Log IoU values for debugging
      // console.log(`${logKey ?? ''} IoU between box ${currentIdx} ${currentBox.score} and box ${idx} ${box.score} = ${o}, ${o <= threshold}`);

      return o <= threshold;
    });
  }

  console.info(`${logKey} After:`, pick)
  return pick;
}

function computeScalePyramid(minFaceSize, width, height) {
  console.info("computeScalePyramid: ", minFaceSize, width, height)
  const m = 12 / minFaceSize;
  const minLayer = Math.min(width, height) * m;

  let scales = [];
  let factor = 0.709; // Scale factor between subsequent scales; adjust as needed
  let scale = m;

  while (minLayer * scale > 12) {
    scales.push(scale);
    scale *= factor;
  }
  console.info("computeScalePyramid scales", scales)
  return scales;
}

function rerec(bbox, maxImageWidth, maxImageHeight) {
  let h = bbox.y2 - bbox.y1;
  let w = bbox.x2 - bbox.x1;
  let l = Math.max(w, h);
  bbox.y1 = Math.round(bbox.y1 + h * 0.5 - l * 0.5);
  bbox.x1 = Math.round(bbox.x1 + w * 0.5 - l * 0.5);
  bbox.y2 = Math.round(bbox.y1 + l);
  bbox.x2 = Math.round(bbox.x1 + l);
  bbox.x1 = Math.max(0, Math.min(bbox.x1, maxImageWidth));
  bbox.y1 = Math.max(0, Math.min(bbox.y1, maxImageHeight));
  bbox.x2 = Math.min(maxImageWidth, bbox.x2);
  bbox.y2 = Math.min(maxImageHeight, bbox.y2);
  bbox.via = "rerec";
  return bbox;
}

function bbreg(originalBox, refinement) {
  let w = originalBox.x2 - originalBox.x1;
  let h = originalBox.y2 - originalBox.y1;

  let b1 = originalBox.x1 + refinement.x1 * w;
  let b2 = originalBox.y1 + refinement.y1 * h;
  let b3 = originalBox.x2 + refinement.x2 * w;
  let b4 = originalBox.y2 + refinement.y2 * h;
  return {
    via: "bbreg",
    x1: Math.round(b1),
    y1: Math.round(b2),
    x2: Math.round(b3),
    y2: Math.round(b4),
    score: originalBox.score
  };
}

// UNUSED:::
function padBoundingBoxes(bbox, maxImageWidth, maxImageHeight) {
  // Compute padding coordinates
  const tmpw = bbox.x2 - bbox.x1 + 1;
  const tmph = bbox.y2 - bbox.y1 + 1;

  let dx = 1, dy = 1, edx = tmpw, edy = tmph;
  let x = bbox.x1, y = bbox.y1, ex = bbox.x2, ey = bbox.y2;

  if (ex > maxImageWidth) {
      edx = -ex + maxImageWidth + tmpw;
      ex = maxImageWidth;
  }
  if (ey > maxImageHeight) {
      edy = -ey + maxImageHeight + tmph;
      ey = maxImageHeight;
  }
  if (x < 1) {
      dx = 2 - x;
      x = 1;
  }
  if (y < 1) {
      dy = 2 - y;
      y = 1;
  }

  return {
      via: "padBoundingBoxes",
      x1: x, y1: y, x2: ex, y2: ey,
      dx, dy, edx, edy,
      score: bbox.score
  };
}