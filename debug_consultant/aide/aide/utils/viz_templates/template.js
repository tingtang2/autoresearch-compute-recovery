// @ts-nocheck
/* eslint-disable */
/* global canvas, createCanvas, windowWidth, windowHeight, mouseX, mouseY, width, height, dist, mouseIsPressed, 
   hljs, awaitingPostResizeOps, resizeCanvas, globalTime, frameRate, frameCount, millis, cos, min, PI, 
   deltaTime, cursor, ARROW, HAND, background, translate, scale, fill, noStroke, square, textAlign, 
   CENTER, textSize, text, stroke, strokeWeight, noFill, bezier, rect, lerpColor, color, bezierPoint */

/*
 * TEMPLATE FILE - NOT VALID JAVASCRIPT UNTIL PROCESSED
 * This file contains <placeholder> tokens that get replaced with actual data
 * All syntax errors are expected and will be resolved during build
 */

const bgCol = "#F2F0E7";
const accentCol = "#fd4578";

let selectedNodeIndex = -1;  // Track which node is currently selected

hljs.initHighlightingOnLoad();

const updateTargetDims = () => {
  // width is max-width of `.contentContainer` - its padding
  // return [min(windowWidth, 900 - 80), 700]
  return [windowWidth * (1 / 2), windowHeight];
};

const setCodeAndPlan = (code, plan) => {
  const codeElm = document.getElementById("code");
  if (codeElm) {
    // codeElm.innerText = code;
    codeElm.innerHTML = hljs.highlight(code, { language: "python" }).value;
  }

  const planElm = document.getElementById("plan");
  if (planElm) {
    // planElm.innerText = plan.trim();
    planElm.innerHTML = hljs.highlight(plan, { language: "plaintext" }).value;
  }
};

function windowResized() {
  resizeCanvas(...updateTargetDims());
  awaitingPostResizeOps = true;
}

const animEase = (t) => 1 - (1 - Math.min(t, 1.0)) ** 5;

// ---- global constants ----

const globalAnimSpeed = 1.1;
const scaleFactor = 0.57;

// ---- global vars ----

let globalTime = 0;
let manualSelection = false;

let currentElemInd = 0;

// eslint-disable-next-line
let treeStructData = /*<placeholder>*/ {};

let lastClick = 0;
let firstFrameTime = undefined;

let nodes = [];
let edges = [];

let lastScrollPos = 0;

const updateSummaryPanel = () => {
  // Update selected nodes
  const summaryList = document.getElementById("summary-list");
  if (summaryList && treeStructData.selected_for_summary) {
    let summaryHtml = "";
    let hasSelected = false;
    
    for (let i = 0; i < treeStructData.selected_for_summary.length; i++) {
      if (treeStructData.selected_for_summary[i]) {
        hasSelected = true;
        const nodeClass = treeStructData.selected_for_summary[i] ? "selected" : "";
        summaryHtml += `<div class="summary-node ${nodeClass}">Node #${i}</div>`;
      }
    }
    
    if (!hasSelected) {
      summaryHtml = "No nodes selected yet...";
    }
    
    summaryList.innerHTML = summaryHtml;
  }
  
  // Selection summary
  const selElm = document.getElementById("selection-summary");
  if (selElm && treeStructData.selection) {
    const sel = treeStructData.selection;
    const rows = [];
    const addRow = (label, obj) => {
      if (!obj) return;
      const dir = obj.maximize === false ? "↓" : "↑";
      rows.push(
        `<tr><td>${label}</td><td>${obj.node ?? "-"}</td><td>${dir} ${obj.cv_mean ?? "-"}</td><td>${obj.cv_std ?? "-"}</td></tr>`
      );
    };
    addRow("Best Raw", sel.best_raw);
    addRow("Mean - k·Std", sel.mean_minus_k_std);
    addRow("Maximin (No Filter)", sel.maximin_no_filter);
    addRow("Post-Search (Config)", sel.post_search);
    let postInfo = "";
    if (sel.post_search && sel.post_search.info) {
      const info = sel.post_search.info;
      const method = info.method ?? "unknown";
      const dir = sel.post_search.maximize === false ? "minimization" : "maximization";

      if (method === "elite_maximin") {
        postInfo = `
          <div class="stat-info">
            <strong>Elite-Maximin (Post-Search):</strong>
            direction=${dir},
            elite_size=${info.elite_size ?? "-"},
            size_topk=${info.size_topk ?? "-"}, size_ratio=${info.size_ratio ?? "-"}, size_stat=${info.size_stat ?? "-"},
            population_mean=${info.population_mean_cv_mean ?? "-"}, population_std=${info.population_std_cv_mean ?? "-"},
	            threshold=${info.threshold ?? "-"}, best_worst_case=${info.best_worst_case ?? "-"},
	            notes=${info.notes ?? "-"}
	          </div>`;
	      } else if (method === "maximin") {
	        postInfo = `
	          <div class="stat-info">
	            <strong>Maximin (Post-Search):</strong>
	            direction=${dir},
	            top_k=${info.top_k ?? "-"}, best_worst_case=${info.best_worst_case ?? "-"}
	          </div>`;
	      } else if (method === "mean_minus_k_std") {
	        postInfo = `
	          <div class="stat-info">
	            <strong>Mean - k·Std (Post-Search):</strong>
	            direction=${dir},
	            k_std=${info.k_std ?? "-"}
	          </div>`;
	      } else {
	        postInfo = `
	          <div class="stat-info">
	            <strong>Post-Search (${method}):</strong>
	            ${JSON.stringify(info)}
	          </div>`;
      }
    }
    selElm.innerHTML = `
      <table class="selection-table">
        <tr><th>Label</th><th>Node</th><th>CV mean</th><th>CV std</th></tr>
        ${rows.join("")}
      </table>
      ${postInfo}
    `;
  }
  
  // Update metrics
  const metricsList = document.getElementById("metrics-list");
  if (metricsList && treeStructData.metric_values) {
    let metricsHtml = "";
    let bestMetric = -1;
    let bestValue = null;

    // Find best metric respecting maximize flag per node
    for (let i = 0; i < treeStructData.metrics.length; i++) {
      if (treeStructData.is_buggy[i]) continue;
      const val = treeStructData.metrics[i];
      const maximize = treeStructData.metric_maximize ? treeStructData.metric_maximize[i] !== false : true;
      if (bestValue === null) {
        bestValue = val;
        bestMetric = i;
        continue;
      }
      if (maximize) {
        if (val > bestValue) {
          bestValue = val;
          bestMetric = i;
        }
      } else {
        if (val < bestValue) {
          bestValue = val;
          bestMetric = i;
        }
      }
    }
    
    for (let i = 0; i < treeStructData.metric_values.length; i++) {
      let metricClass = "";
      if (treeStructData.is_buggy && treeStructData.is_buggy[i]) {
        metricClass = "buggy";
      } else if (i === bestMetric) {
        metricClass = "best";
      }
      
      metricsHtml += `<div class="metric-item ${metricClass}">Node #${i}: ${treeStructData.metric_values[i]}</div>`;
    }
    
    if (metricsHtml === "") {
      metricsHtml = "No metrics available...";
    }
    
    metricsList.innerHTML = metricsHtml;
  }
};

const updateDynamicPanel = (nodeIndex) => {
  const summaryList = document.getElementById("summary-list");
  
  if (nodeIndex === -1 || !summaryList) {
    if (summaryList) {
      summaryList.innerHTML = "Click a node to see which nodes it referenced";
    }
    return;
  }
  
  const parentNodeId = (() => {
    if (!treeStructData.edges) return null;
    for (const edge of treeStructData.edges) {
      if (edge[1] === nodeIndex) return edge[0];
    }
    return null;
  })();

  // Show which nodes this selected node "saw"
  const seenNodes = treeStructData.seen_nodes_per_node ? treeStructData.seen_nodes_per_node[nodeIndex] : [];
  
  let summaryHTML = `<div style="font-weight: bold; margin-bottom: 10px; font-size: 14px;">Node #${nodeIndex} Details</div>`;
  summaryHTML += `<div style="margin-bottom: 10px;">Parent: ${parentNodeId === null ? "None (root)" : `Node #${parentNodeId}`}</div>`;
  summaryHTML += `<div style="font-weight: bold; margin-bottom: 10px; font-size: 14px;">Referenced:</div>`;
  if (!seenNodes || seenNodes.length === 0) {
    summaryHTML += `<div style="padding: 0.5em;">No referenced nodes</div>`;
  } else {
    summaryHTML += '<div>';
    
    // Find best metric among seen nodes
    let bestSeenMetric = -Infinity;
    let bestSeenNodeId = -1;
    seenNodes.forEach(seenNodeId => {
      if (!treeStructData.is_buggy[seenNodeId] && treeStructData.metrics[seenNodeId] > bestSeenMetric) {
        bestSeenMetric = treeStructData.metrics[seenNodeId];
        bestSeenNodeId = seenNodeId;
      }
    });
    
    seenNodes.forEach(seenNodeId => {
      const seenMetric = treeStructData.metric_values[seenNodeId];
      const seenIsBuggy = treeStructData.is_buggy[seenNodeId];
      
      let nodeClass = "summary-node";
      if (seenIsBuggy) {
        nodeClass += " buggy";
      } else if (seenNodeId === bestSeenNodeId) {
        nodeClass += " best";
      }
      
      summaryHTML += `<div class="${nodeClass}">Node #${seenNodeId}: ${seenMetric}</div>`;
    });
    summaryHTML += '</div>';
  }
  
  summaryList.innerHTML = summaryHTML;
};

function setup() {
  const canvasContainer = document.getElementById("canvas-container");
  canvas = createCanvas(...updateTargetDims());
  if (canvasContainer) {
    canvasContainer.appendChild(canvas.canvas);
  }
  updateSummaryPanel();
  updateDynamicPanel(-1);
}

class Node {
  constructor(x, y, relSize, treeInd, isSelectedForSummary = false) {
    const minSize = 35;
    const maxSize = 60;

    const maxColor = 10;
    const minColor = 125;

    // Initialize all properties in constructor
    this.x = x;
    this.y = y;
    this.size = minSize + (maxSize - minSize) * relSize;
    this.xT = x;
    this.yT = y - this.size / 2;
    this.xB = x;
    this.yB = y + this.size / 2;
    this.treeInd = treeInd;
    this.color = minColor + (maxColor - minColor) * relSize;
    this.relSize = relSize;
    this.animationStart = Number.MAX_VALUE;
    this.animationProgress = 0;
    this.isStatic = false;
    this.hasChildren = false;
    this.isRootNode = true;
    this.isStarred = false;
    this.selected = false;
    this.renderSize = 10;
    this.edges = [];
    this.bgCol = Math.round(Math.max(this.color / 2, 0));
    this.isSelectedForSummary = isSelectedForSummary;

    nodes.push(this);
  }

  startAnimation(offset = 0) {
    if (this.animationStart == Number.MAX_VALUE)
      this.animationStart = globalTime + offset;
  }

  child(node) {
    let edge = new Edge(this, node);
    this.edges.push(edge);
    edges.push(edge);
    this.hasChildren = true;
    node.isRootNode = false;
    return node;
  }

  render() {
    if (globalTime - this.animationStart < 0) return;

    const mouseXlocalCoords = (mouseX - width / 2) / scaleFactor;
    const mouseYlocalCoords = (mouseY - height / 2) / scaleFactor;
    const isMouseOver =
      dist(mouseXlocalCoords, mouseYlocalCoords, this.x, this.y) <
      this.renderSize / 1.5;
    if (isMouseOver) cursor(HAND);
    if (isMouseOver && mouseIsPressed) {
      nodes.forEach((n) => (n.selected = false));
      this.selected = true;
      selectedNodeIndex = this.treeInd;
      setCodeAndPlan(
        treeStructData.code[this.treeInd],
        treeStructData.plan[this.treeInd],
      );
      updateDynamicPanel(this.treeInd);
      manualSelection = true;
    }

    this.renderSize = this.size;
    if (!this.isStatic) {
      this.animationProgress = animEase(
        (globalTime - this.animationStart) / 1000,
      );
      if (this.animationProgress >= 1) {
        this.isStatic = true;
      } else {
        this.renderSize =
          this.size *
          (0.8 +
            0.2 *
              (-3.33 * this.animationProgress ** 2 +
                4.33 * this.animationProgress));
      }
    }

    fill(this.color);
    if (this.selected) {
      fill(accentCol);
    }
    if (this.isSelectedForSummary) {
      fill("#FFD700"); // Gold color for nodes selected for summary
    }

    noStroke();
    square(
      this.x - this.renderSize / 2,
      this.y - this.renderSize / 2,
      this.renderSize,
      10,
    );

    noStroke();
    textAlign(CENTER, CENTER);
    textSize(this.renderSize / 2);
    fill(255);
    // Show node number instead of "{ }"
    text("#" + this.treeInd, this.x, this.y - 1);

    const dotAnimThreshold = 0.85;
    if (this.isStarred && this.animationProgress >= dotAnimThreshold) {
      let dotAnimProgress =
        (this.animationProgress - dotAnimThreshold) / (1 - dotAnimThreshold);
      textSize(
        ((-3.33 * dotAnimProgress ** 2 + 4.33 * dotAnimProgress) *
          this.renderSize) /
          2,
      );
      if (this.selected) {
        fill(0);
        stroke(0);
      } else {
        fill(accentCol);
        stroke(accentCol);
      }
      strokeWeight((-(dotAnimProgress ** 2) + dotAnimProgress) * 2);
      text("*", this.x + 20, this.y - 11);
      noStroke();
    }

    if (!this.isStatic) {
      fill(bgCol);
      const progressAnimBaseSize = this.renderSize + 5;
      rect(
        this.x - progressAnimBaseSize / 2,
        this.y -
          progressAnimBaseSize / 2 +
          progressAnimBaseSize * this.animationProgress,
        progressAnimBaseSize,
        progressAnimBaseSize * (1 - this.animationProgress),
      );
    }
    if (this.animationProgress >= 0.9) {
      this.edges
        .sort((a, b) => a.color() - b.color())
        .forEach((e, i) => {
          e.startAnimation((i / this.edges.length) ** 2 * 1000);
        });
    }
  }
}

class Edge {
  constructor(nodeT, nodeB) {
    this.nodeT = nodeT;
    this.nodeB = nodeB;
    this.animX = 0;
    this.animY = 0;
    this.animationStart = Number.MAX_VALUE;
    this.animationProgress = 0;
    this.isStatic = false;
    this.weight = 2 + nodeB.relSize * 1;
  }

  color() {
    return this.nodeB.color;
  }

  startAnimation(offset = 0) {
    if (this.animationStart == Number.MAX_VALUE)
      this.animationStart = globalTime + offset;
  }

  render() {
    if (globalTime - this.animationStart < 0) return;

    if (!this.isStatic) {
      this.animationProgress = animEase(
        (globalTime - this.animationStart) / 1000,
      );
      if (this.animationProgress >= 1) {
        this.isStatic = true;
        this.animX = this.nodeB.xT;
        this.animY = this.nodeB.yT;
      } else {
        this.animX = bezierPoint(
          this.nodeT.xB,
          this.nodeT.xB,
          this.nodeB.xT,
          this.nodeB.xT,
          this.animationProgress,
        );

        this.animY = bezierPoint(
          this.nodeT.yB,
          (this.nodeT.yB + this.nodeB.yT) / 2,
          (this.nodeT.yB + this.nodeB.yT) / 2,
          this.nodeB.yT,
          this.animationProgress,
        );
      }
    }
    if (this.animationProgress >= 0.97) {
      this.nodeB.startAnimation();
    }

    strokeWeight(this.weight);
    noFill();
    stroke(
      lerpColor(color(bgCol), color(accentCol), this.nodeB.relSize * 1 + 0.7),
    );
    bezier(
      this.nodeT.xB,
      this.nodeT.yB,
      this.nodeT.xB,
      (this.nodeT.yB + this.nodeB.yT) / 2,
      this.animX,
      (this.nodeT.yB + this.nodeB.yT) / 2,
      this.animX,
      this.animY,
    );
  }
}

function draw() {
  cursor(ARROW);
  frameRate(120);
  if (!firstFrameTime && frameCount <= 1) {
    firstFrameTime = millis();
  }
  // ---- update global animation state ----
  const initialSpeedScalingEaseIO =
    (cos(min((millis() - firstFrameTime) / 8000, 1.0) * PI) + 1) / 2;
  const initialSpeedScalingEase =
    (cos(min((millis() - firstFrameTime) / 8000, 1.0) ** (1 / 2) * PI) + 1) / 2;
  const initAnimationSpeedFactor = 1.0 - 0.4 * initialSpeedScalingEaseIO;
  // update global scaling-aware clock
  globalTime += globalAnimSpeed * initAnimationSpeedFactor * deltaTime;

  if (nodes.length == 0) {
    const spacingHeight = height * 1.3;
    const spacingWidth = width * 1.3;
    treeStructData.layout.forEach((lay, index) => {
      new Node(
        spacingWidth * lay[0] - spacingWidth / 2,
        20 + spacingHeight * lay[1] - spacingHeight / 2,
        1 - treeStructData.metrics[index],
        index,
        treeStructData.selected_for_summary ? treeStructData.selected_for_summary[index] : false,
      );
    });
    treeStructData.edges.forEach((ind) => {
      nodes[ind[0]].child(nodes[ind[1]]);
    });
    nodes.forEach((n) => {
      if (n.isRootNode) n.startAnimation();
    });
    nodes[0].selected = true;
    setCodeAndPlan(
      treeStructData.code[0],
      treeStructData.plan[0],
    );
    updateSummaryPanel();
  }

  const staticNodes = nodes.filter(
    (n) => n.isStatic || n.animationProgress >= 0.7,
  );
  if (staticNodes.length > 0) {
    const largestNode = staticNodes.reduce((prev, current) =>
      prev.relSize > current.relSize ? prev : current,
    );
    if (!manualSelection) {
      if (!largestNode.selected) {
        setCodeAndPlan(
          treeStructData.code[largestNode.treeInd],
          treeStructData.plan[largestNode.treeInd],
        );
      }
      staticNodes.forEach((node) => {
        node.selected = node === largestNode;
      });
    }
  }
  background(bgCol);
  // global animation transforms
  translate(width / 2, height / 2);
  scale(scaleFactor);

  
  // ---- fg render ----
  edges.forEach((e) => e.render());
  nodes.forEach((n) => n.render());
  
};
