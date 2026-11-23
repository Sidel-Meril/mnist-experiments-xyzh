const NAMES = {
  realnvp: "Norm Flow (RealNVP)",
  ae: "Autoencoder Compression",
  vae: "Variational Autoencoder (VAE)",
  fvae: "f-VAE",
  betavae: "β-VAE Exploration",
  gan: "Vanilla GAN",
  cgan: "Conditional and Advanced GAN Variants",
  pgan: "Progressive Growing GAN",
};

const QUESTIONS = [
  ["worked", "What worked well?"],
  ["unexpected", "What didn’t turn out as expected?"],
  ["why", "Why do you think the results might not be great?"],
  ["improve", "How could the results be improved?"],
];

function frames(dir, from, to, text) {
  const list = [];
  for (let epoch = from; epoch <= to; epoch += 1) {
    const name = String(epoch).padStart(4, "0");
    list.push({
      epoch,
      src: `${dir}/epoch_${name}.png`,
      caption: text(epoch),
    });
  }
  return list;
}

function block(parent, id, title, wide) {
  const section = el("section", wide ? "plot-block wide" : "plot-block");
  section.id = id;
  section.append(el("h4", null, title));
  parent.append(section);
  return section;
}

function modeling(article, bundle) {
  const section = el("section", "modeling");
  section.append(el("h3", null, "Modeling"));
  const blocks = Array.isArray(bundle.process) ? bundle.process : [bundle.process];
  blocks.forEach((text) => section.append(el("p", "process", text)));
  article.append(section);
}

function discuss(article, paragraphs, extras) {
  const section = el("section", "discussion");
  section.append(el("h3", null, "Discussion"));
  QUESTIONS.forEach(([key, title]) => {
    const item = el("section");
    item.append(el("h4", null, title));
    item.append(paragraphs[key]);
    section.append(item);
  });
  extras.forEach(([title, body]) => {
    const item = el("section", "extra");
    item.append(el("h4", null, title));
    body.forEach((node) => item.append(node));
    section.append(item);
  });
  article.append(section);
}

function sources(article, bundle) {
  const line = el("p", "sources");
  line.append(document.createTextNode("Sources: "));
  bundle.sources.forEach((source, index) => {
    if (index) line.append(document.createTextNode(" · "));
    line.append(el("code", null, source));
  });
  article.append(line);
}

function chartOf(bundle, title) {
  return bundle.charts.find((chart) => chart.title === title);
}

const PGAN = [
  { epoch: 1, train_d: 1.5568, train_g: 0.6524, eval_d: 1.4584, eval_g: 0.7736, sample_std: 0.5809, resolution: 4 },
  { epoch: 2, train_d: 1.4081, train_g: 0.8409, eval_d: 1.4031, eval_g: 0.8568, sample_std: 0.6119, resolution: 4 },
  { epoch: 3, train_d: 1.3805, train_g: 0.7972, eval_d: 1.3672, eval_g: 0.7585, sample_std: 0.4146, resolution: 8 },
  { epoch: 4, train_d: 1.2115, train_g: 1.0065, eval_d: 1.132, eval_g: 1.0709, sample_std: 0.4054, resolution: 8 },
  { epoch: 5, train_d: 0.919, train_g: 1.3526, eval_d: 0.9453, eval_g: 1.0922, sample_std: 0.5039, resolution: 8 },
  { epoch: 6, train_d: 1.0659, train_g: 1.2692, eval_d: 0.7338, eval_g: 1.786, sample_std: 0.4142, resolution: 8 },
  { epoch: 7, train_d: 1.3452, train_g: 0.7072, eval_d: 1.3287, eval_g: 0.7126, sample_std: 0.2735, resolution: 16 },
  { epoch: 8, train_d: 1.163, train_g: 1.0645, eval_d: 1.1019, eval_g: 0.8897, sample_std: 0.4495, resolution: 16 },
  { epoch: 9, train_d: 1.0269, train_g: 1.1127, eval_d: 1.0425, eval_g: 1.1904, sample_std: 0.4057, resolution: 16 },
  { epoch: 10, train_d: 1.0977, train_g: 0.9943, eval_d: 1.094, eval_g: 1.2155, sample_std: 0.3316, resolution: 16 },
  { epoch: 11, train_d: 1.2302, train_g: 0.7011, eval_d: 1.0976, eval_g: 0.7598, sample_std: 0.2165, resolution: 32 },
  { epoch: 12, train_d: 1.1058, train_g: 1.0312, eval_d: 1.1703, eval_g: 0.7478, sample_std: 0.2047, resolution: 32 },
  { epoch: 13, train_d: 0.8979, train_g: 1.0221, eval_d: 0.6342, eval_g: 1.4033, sample_std: 0.1307, resolution: 32 },
  { epoch: 14, train_d: 0.7982, train_g: 1.3313, eval_d: 1.5197, eval_g: 0.6999, sample_std: 0.1041, resolution: 32 },
];

const PGAN_FILES = [
  [1, "epoch_0001_res04_stable.png", "stable 4×4"],
  [2, "epoch_0002_res04_stable.png", "stable 4×4"],
  [3, "epoch_0003_res08_fade.png", "fade 8×8"],
  [4, "epoch_0004_res08_fade.png", "fade 8×8"],
  [5, "epoch_0005_res08_stable.png", "stable 8×8"],
  [6, "epoch_0006_res08_stable.png", "stable 8×8"],
  [7, "epoch_0007_res16_fade.png", "fade 16×16"],
  [8, "epoch_0008_res16_fade.png", "fade 16×16"],
  [9, "epoch_0009_res16_stable.png", "stable 16×16"],
  [10, "epoch_0010_res16_stable.png", "stable 16×16"],
  [11, "epoch_0011_res32_fade.png", "fade 32×32"],
  [12, "epoch_0012_res32_fade.png", "fade 32×32"],
  [13, "epoch_0013_res32_stable.png", "stable 32×32"],
  [14, "epoch_0014_res32_stable.png", "stable 32×32"],
];

const TRAVERSAL = [
  [0.0507, 0.0471, 0.0521, 0.0615, 0.041, 0.0366, 0.0528, 0.0366],
  [0.0673, 0.0572, 0.0804, 0.096, 0.065, 0.0478, 0.0718, 0.0673],
  [0.066, 0.047, 0.0648, 0.0755, 0.0005, 0.0283, 0.1054, 0.0614],
  [0.057, 0.022, 0.0397, 0.002, 0.0009, 0.0008, 0.0949, 0.0693],
];

const SPREAD = {
  digits: [
    { label: "Vanilla", value: 0.0354 },
    { label: "cGAN", value: 1.047, kept: true },
    { label: "Patch", value: 0.883 },
    { label: "FM", value: 0.926 },
  ],
  fashion: [
    { label: "Vanilla", value: 0.0778 },
    { label: "cGAN", value: 0.949, kept: true },
    { label: "Patch", value: 1.101 },
    { label: "FM", value: 1.124 },
  ],
};

function pair(train, test, trainName, testName) {
  return [
    { name: trainName || "Train", value: train },
    { name: testName || "Test", value: test },
  ];
}

function renderRealnvp(article, bundle) {
  modeling(article, bundle);
  const plots = el("section", "plots");
  plots.append(el("h3", null, "Plots"));
  const grid = el("div", "plot-grid");

  const gen = block(grid, "realnvp-gen", "Generation per epoch", true);
  const genBody = el("div");
  const genState = { dataset: "digits" };
  const genSets = {
    digits: frames("figures/realnvp/digits", 1, 20, (epoch) => `Digit samples from the inverse, epoch ${epoch}.`),
    fashion: frames("figures/realnvp/fashion", 1, 12, (epoch) => `Fashion-MNIST samples from the inverse, epoch ${epoch}.`),
  };
  switchBar(gen, "Dataset", [
    { id: "digits", label: "Digits" },
    { id: "fashion", label: "Fashion" },
  ], genState.dataset, (id) => {
    genState.dataset = id;
    mountSet();
  });
  gen.append(genBody);
  let stopGen = () => {};
  function mountSet() {
    stopGen();
    genBody.replaceChildren();
    stopGen = mountReel(genBody, genSets[genState.dataset]);
  }
  mountSet();

  const loss = block(grid, "realnvp-loss", "Loss");
  const lossBody = el("div");
  const bpd = chartOf(bundle, "Fashion-MNIST test bits per dimension").marks.map((mark) => ({
    epoch: mark.epoch,
    bpd: mark.bpd,
  }));
  const lowest = bpd.reduce((best, row) => (row.bpd < best.bpd ? row : best));
  switchBar(loss, "Dataset", [
    { id: "digits", label: "Digits" },
    { id: "fashion", label: "Fashion" },
  ], "fashion", (id) => showLoss(id));
  loss.append(lossBody);
  function showLoss(id) {
    lossBody.replaceChildren();
    if (id === "digits") {
      still(
        lossBody,
        "figures/realnvp/digits/bpd.png",
        "Digit bits per dimension. The test line ends at 1.210.",
        "Digits. The per-epoch table was not saved. This is the stored curve. Test bits per dimension end at 1.210.",
      );
      return;
    }
    mountRibbon(lossBody, bpd, [{ key: "bpd", label: "Test bpd" }], 20);
    caption(lossBody, `Fashion-MNIST test bits per dimension. The bold bar is epoch 20, the checkpoint in the test file. The lowest point on the curve is epoch ${lowest.epoch} at ${fmt(lowest.bpd)}.`);
  }
  showLoss("fashion");

  const search = block(grid, "realnvp-optuna", "Optuna");
  mountStems(search, [
    { label: "64 ch", value: 860.1, kept: true },
    { label: "32 ch", value: 1050.1 },
  ]);
  caption(search, "Validation negative log-likelihood on digits, one epoch, two folds. The kept trial is 64 hidden channels, scale 2, learning rate 0.001, batch 128. Fashion reused these settings, so there is one search.");

  const bars = block(grid, "realnvp-bars", "Train and test");
  const barBody = el("div");
  const splits = {
    digits: [
      { label: "Bits / dim", bars: pair(1.215, 1.21) },
      { label: "NLL", bars: pair(660.3, 657.3) },
    ],
    fashion: [
      { label: "Bits / dim", bars: pair(3.245, 3.263) },
      { label: "NLL", bars: pair(1763.6, 1773.2) },
    ],
  };
  switchBar(bars, "Dataset", [
    { id: "digits", label: "Digits" },
    { id: "fashion", label: "Fashion" },
  ], "digits", (id) => {
    barBody.replaceChildren();
    mountPairs(barBody, splits[id]);
  });
  bars.append(barBody);
  mountPairs(barBody, splits.digits);
  caption(bars, "Train beside test. Bar length uses each metric’s own scale. The zero line sits at the top when a score is negative.");

  plots.append(grid);
  article.append(plots);

  discuss(article, {
    worked: linkPara([
      "Digit test likelihood lands at 1.210 bits per dimension, and train is 1.215. Fashion falls from 3.582 to 3.263. Encode-then-invert keeps the stroke and the clothes. The wider trial wins, 860.1 against 1050.1. See the ",
      { href: "#realnvp-loss", label: "likelihood curve" },
      " and the ",
      { href: "#realnvp-optuna", label: "search" },
      ".",
    ]),
    unexpected: linkPara([
      "Draws from the prior are torn on both datasets, while the reconstruction of a real image stays sharp. Fashion levels off near 3.26. See the ",
      { href: "#realnvp-gen", label: "epoch reel" },
      ".",
    ]),
    why: linkPara([
      "The search is two trials of one epoch, and Fashion inherited the digit settings. A density can explain the empty background and still miss the glyph when noise is inverted. See the ",
      { href: "#realnvp-bars", label: "train and test bars" },
      ".",
    ]),
    improve: linkPara([
      "Give Fashion its own search and a longer run. If the curve is flat and the prior samples stay broken, widen the coupling networks. The ",
      { href: "#realnvp-loss", label: "Fashion curve" },
      " is already on that plateau.",
    ]),
  }, [
    ["Coupling layer and multi-scale", [
      el("p", null, "CouplingLayer in src/mnist_realnvp/model.py keeps every coordinate where the mask is 1. Where the mask is 0, forward writes y = mask * x + (1 - mask) * (x * exp(log_s) + shift), and the log-determinant is the sum of log_s on those changed coordinates only. log_s is scale * tanh(raw_s), so the scale stays bounded. The inverse recomputes the same scale and shift from the kept half."),
      el("p", null, "MultiScaleFlow.forward squeezes 28×28 into 14×14, runs the first coupling block, and factors z1 off as 2×14×14. It squeezes again to 7×7, runs the second block, factors z2, and the last block emits z3. Both leftovers are 4×7×7. Each piece is scored as a standard normal, and the logit Jacobian of the pixels is added to the flow Jacobian."),
    ]],
    ["Likelihood", [
      linkPara([
        "1.210 bits per dimension on the 10,000 digit test images is a reasonable likelihood for this small flow. 3.263 on Fashion is a partial fit. The bold bar on the ",
        { href: "#realnvp-loss", label: "Fashion curve" },
        ` is epoch 20 because that is the checkpoint written to the test file after a fixed 20-epoch run. The curve has already flattened: the lowest saved point is epoch ${lowest.epoch} at ${fmt(lowest.bpd)}, and epoch 20 is 3.263. Epoch 20 is the one to cite because it was scored, and it sits on that plateau.`,
      ]),
    ]],
    ["How the search chose", [
      linkPara([
        "The objective was validation negative log-likelihood, the same loss the flow trains, so the search ranks density rather than how a sample looks. Two trials, one epoch, two folds. The kept trial uses 64 hidden channels, scale 2, learning rate 0.001, and batch 128: 860.1 nats, 1.583 bits per dimension. The other uses 32 channels, scale 1, learning rate 0.0001, weight decay 1e-6, and batch 64: 1050.1 nats, 1.932 bits. The wider net and the larger step won. See ",
        { href: "#realnvp-optuna", label: "the two trials" },
        ".",
      ]),
    ]],
    ["What a reasonable sample was", [
      linkPara([
        "A reasonable draw would be a digit or a garment you can name. Reconstructions meet that. Prior samples do not: broken hooks, torn cloth. The run already tried the fix that matches the likelihood: a separate digit checkpoint, and the wider coupling net from the search. That raised the density and repaired encode-and-invert. It left prior noise off the glyph. The ",
        { href: "#realnvp-gen", label: "reel" },
        " is that gap. A Fashion-specific search is the next lever, because this Fashion run used the digit hyperparameters.",
      ]),
    ]],
  ]);
}

function renderAe(article, bundle) {
  modeling(article, bundle);
  const plots = el("section", "plots");
  plots.append(el("h3", null, "Plots"));
  const grid = el("div", "plot-grid");
  const marks = chartOf(bundle, "Validation MSE by latent size").marks;

  const gen = block(grid, "ae-gen", "Generation per epoch", true);
  mountReel(gen, frames("figures/ae", 1, 8, (epoch) => `Autoencoder reconstruction, epoch ${epoch}. Original digits on the top row. Digits only.`));

  const loss = block(grid, "ae-loss", "Loss");
  still(loss, "figures/ae/loss.png", "Reconstruction MSE for the latent-128 checkpoint.", "Training reconstruction MSE for the latent-128 run. The test curve ends at 0.041, the scored test error. A numeric per-epoch table was not saved, so this is the stored figure.");

  const search = block(grid, "ae-optuna", "Optuna");
  mountStems(search, marks.map((mark) => ({
    label: String(mark.latent),
    value: mark.val_mse,
    kept: mark.latent === 128,
  })));
  const drops = [];
  for (let i = 1; i < marks.length; i += 1) {
    drops.push({
      label: String(marks[i].latent),
      value: marks[i - 1].val_mse - marks[i].val_mse,
      kept: marks[i].latent === 16,
      tag: "knee",
    });
  }
  mountStems(search, drops);
  caption(search, "Top: validation MSE against latent size. The solid mark is 128, the lowest point and the scored code. Bottom: how much MSE fell when the code doubled. The solid mark is 16, where the drop becomes small. The Optuna database has no finished trial, so hidden size, dropout, and learning rate were never scored. Digits only.");

  const bars = block(grid, "ae-bars", "Train and test");
  mountPairs(bars, [
    { label: "MSE", bars: pair(0.04137, 0.04135) },
    { label: "MAE", bars: pair(0.07726, 0.07708) },
  ]);
  caption(bars, "Train and test for the latent-128 checkpoint. They match. Compression is 784 / 128 = 6.125. The validation sweep on the Optuna chart is a separate measurement and sits near 0.096.");

  plots.append(grid);
  article.append(plots);

  discuss(article, {
    worked: linkPara([
      "The epoch grids go from a blur to a readable digit, and the latent-128 checkpoint copies a test digit at MSE 0.0414 and MAE 0.0771. See the ",
      { href: "#ae-gen", label: "epoch reel" },
      " and the ",
      { href: "#ae-loss", label: "training curve" },
      ".",
    ]),
    unexpected: linkPara([
      "Validation MSE only moves from 0.0975 at latent 64 to 0.0964 at 128. The hyperparameter search recorded no finished trial. See the ",
      { href: "#ae-optuna", label: "latent chart" },
      ".",
    ]),
    why: linkPara([
      "Almost all of the drop is spent by latent size 16, from 0.1867 at size 2 to 0.1042. A digit stroke fits in a short code. The scored run then reaches test MSE 0.0414, and train matches it. See the ",
      { href: "#ae-bars", label: "bars" },
      ". The validation sweep on the latent chart is a higher, earlier measurement.",
    ]),
    improve: linkPara([
      "For compression, stop near 16 or 32. Finish the search before treating any other setting as tested. A sharper copy needs a different loss. The marginal chart in ",
      { href: "#ae-optuna", label: "Optuna" },
      " is the reason to stop early.",
    ]),
  }, [
    ["Training, compression, and reconstruction", [
      linkPara([
        "The ",
        { href: "#ae-loss", label: "loss figure" },
        " is the latent-128 run: reconstruction MSE falls and the test line ends at 0.041. The ",
        { href: "#ae-gen", label: "reel" },
        " shows the same thing in pixels across eight epochs. The original row is recognizable, and the copy softens a 0 and a 4 while keeping the class.",
      ]),
    ]],
    ["Latent size", [
      el("p", null, "The saved validation sweep is the comparison we have: latent 2 → 0.1867, 4 → 0.1496, 8 → 0.1154, 16 → 0.1042, 32 → 0.0999, 64 → 0.0975, 128 → 0.0964. The second stem chart is the drop at each doubling: about 0.037, then 0.034, then 0.011, then under 0.005. Past 16 the extra coordinates store very little error. best_params and the training note agree that the Optuna study finished no trial, so this sweep is a latent-size curve, not a completed search over hidden size or learning rate."),
      linkPara([
        "The scored model uses 128 because that point is the minimum on the curve. The gain from 32 to 128 is 0.0035 validation MSE. A shorter code would have been the better compression. The ",
        { href: "#ae-bars", label: "bars" },
        " show train MSE 0.0414 beside test 0.0414, so that final fit generalized. The 0.0964 on the latent chart is the validation sweep, a different stage.",
      ]),
    ]],
  ]);
}

function renderVae(article, bundle) {
  modeling(article, bundle);
  const plots = el("section", "plots");
  plots.append(el("h3", null, "Plots"));
  const grid = el("div", "plot-grid");

  const gen = block(grid, "vae-gen", "Generation per epoch", true);
  const genBody = el("div");
  let stopGen = () => {};
  switchBar(gen, "Dataset", [
    { id: "digits", label: "Digits" },
    { id: "fashion", label: "Fashion" },
  ], "digits", (id) => mountGen(id));
  gen.append(genBody);
  function mountGen(id) {
    stopGen();
    genBody.replaceChildren();
    const dir = id === "digits" ? "figures/vae/digits" : "figures/vae/fashion";
    const name = id === "digits" ? "Digits" : "Fashion-MNIST";
    stopGen = mountReel(genBody, frames(dir, 1, 20, (epoch) => `${name} VAE samples, epoch ${epoch}.`));
  }
  mountGen("digits");

  const loss = block(grid, "vae-loss", "Loss");
  const lossBody = el("div");
  switchBar(loss, "Dataset", [
    { id: "digits", label: "Digits" },
    { id: "fashion", label: "Fashion" },
  ], "digits", (id) => showLoss(id));
  loss.append(lossBody);
  function showLoss(id) {
    lossBody.replaceChildren();
    const src = id === "digits" ? "figures/vae/digits/loss.png" : "figures/vae/fashion/loss.png";
    const text = id === "digits"
      ? "Digit VAE loss. The test line falls from about 137 to about 86 and then flattens. The per-epoch numbers were saved as this figure."
      : "Fashion-MNIST VAE loss. The test line ends near 74, under the digit curve.";
    still(lossBody, src, text, text);
  }
  showLoss("digits");

  const search = block(grid, "vae-optuna", "Optuna");
  const latent = chartOf(bundle, "Validation loss by latent size").marks;
  mountStems(search, latent.map((mark) => ({
    label: String(mark.latent),
    value: mark.val_loss,
    kept: !!mark.selected,
  })));
  caption(search, "Validation loss on the short search. The objective is that loss, so latent size 8 is kept at 95.9. Latent 16 scores 101.4 and latent 32 scores 107.4. Both datasets were then trained at latent size 8. One search, so no dataset switch.");

  const bars = block(grid, "vae-bars", "Train and test");
  const barBody = el("div");
  const splits = {
    digits: [
      { label: "MSE", bars: pair(0.0812, 0.0852) },
      { label: "ELBO", bars: pair(-78.97, -82.21) },
      { label: "KL", bars: pair(15.34, 15.41) },
      { label: "KID", bars: [{ name: "Test", value: 3.105 }] },
    ],
    fashion: [
      { label: "MSE", bars: pair(0.0724, 0.0749) },
      { label: "ELBO", bars: pair(-69.21, -71.09) },
      { label: "KL", bars: pair(12.42, 12.34) },
      { label: "KID", bars: [{ name: "Test", value: 0.204 }] },
    ],
  };
  switchBar(bars, "Dataset", [
    { id: "digits", label: "Digits" },
    { id: "fashion", label: "Fashion" },
  ], "digits", (id) => {
    barBody.replaceChildren();
    mountPairs(barBody, splits[id]);
  });
  bars.append(barBody);
  mountPairs(barBody, splits.digits);
  caption(bars, "Same latent size, 1,000 images per class. ELBO bars hang below zero. KID is test only.");

  plots.append(grid);
  article.append(plots);

  discuss(article, {
    worked: linkPara([
      "Latent size 8 wins the four-trial search at validation loss 95.9. Both sample grids are readable, and Fashion beats digits on MSE, ELBO, KL, and KID. See the ",
      { href: "#vae-optuna", label: "search" },
      " and the ",
      { href: "#vae-bars", label: "bars" },
      ".",
    ]),
    unexpected: linkPara([
      "Latent sizes 16 and 32 score worse on this short search, 101.4 and 107.4. Digit samples look readable and still sit at KID 3.11, against 0.20 on Fashion. See the ",
      { href: "#vae-gen", label: "reel" },
      ".",
    ]),
    why: linkPara([
      "Both scored runs use 1,000 images per class, and the per-epoch record is the loss figure. Blur removes a one-pixel stroke and still leaves a shirt, which is why KID separates the datasets more than MSE. See the ",
      { href: "#vae-loss", label: "loss figures" },
      ".",
    ]),
    improve: linkPara([
      "Retrain both models on the full 60,000 images and write the per-epoch ELBO next to the final table. This slice and the full-split f-VAE are different experiments. The ",
      { href: "#vae-bars", label: "bars" },
      " are the slice.",
    ]),
  }, [
    ["Reconstruction and likelihood", [
      el("p", null, "Reconstruction is reasonable and soft. On digits the class survives and the 8 breaks. On Fashion the silhouette survives and the print, the sandal, and the lamp do not. The ELBO is a real bound: digits test −82.21 with train −78.97, Fashion test −71.09 with train −69.21. Train sits next to test on both, so the bound held on the slice. Fashion is ahead on every saved score, including KID 0.204 against 3.105."),
    ]],
    ["Generation against RealNVP", [
      linkPara([
        "Draws from the prior can be named on both datasets. The RealNVP inverse grids are torn hooks and torn cloth. On generation, this VAE is ahead of that flow. The likelihoods are different objects: RealNVP reports bits per dimension of an exact density on the full digit set, 1.210, and this ELBO is a Gaussian-decoder bound on a 10,000-image slice. The ",
        { href: "#vae-gen", label: "reel" },
        " is the generation comparison. The files support that comparison, and they are the wrong pair of files for a likelihood ranking.",
      ]),
    ]],
  ]);
}

function renderFvae(article, bundle) {
  modeling(article, bundle);
  const plots = el("section", "plots");
  plots.append(el("h3", null, "Plots"));
  const grid = el("div", "plot-grid");

  const gen = block(grid, "fvae-gen", "Generation per epoch", true);
  const genBody = el("div");
  let stopGen = () => {};
  switchBar(gen, "Dataset", [
    { id: "digits", label: "Digits" },
    { id: "fashion", label: "Fashion" },
  ], "digits", (id) => mountGen(id));
  gen.append(genBody);
  function mountGen(id) {
    stopGen();
    genBody.replaceChildren();
    if (id === "digits") {
      stopGen = mountReel(genBody, frames("figures/fvae", 1, 14, (epoch) => `f-VAE digits, epoch ${epoch}.`));
      return;
    }
    stopGen = mountReel(genBody, frames("figures/fvae/fashion", 1, 20, (epoch) => `f-VAE Fashion-MNIST, epoch ${epoch}.`));
  }
  mountGen("digits");

  const loss = block(grid, "fvae-loss", "Loss");
  const lossBody = el("div");
  switchBar(loss, "Dataset", [
    { id: "digits", label: "Digits" },
    { id: "fashion", label: "Fashion" },
  ], "digits", (id) => showLoss(id));
  loss.append(lossBody);
  function showLoss(id) {
    lossBody.replaceChildren();
    const src = id === "digits" ? "figures/fvae/digits/loss.png" : "figures/fvae/fashion/loss.png";
    const text = id === "digits"
      ? "f-VAE digit loss on the full split. Train and test stay together. The per-epoch table was saved as this figure."
      : "f-VAE Fashion-MNIST loss through epoch 20. The test line is still falling at the end.";
    still(lossBody, src, text, text);
  }
  showLoss("digits");

  const search = block(grid, "fvae-optuna", "Optuna");
  mountStems(search, [
    { label: "Digits", value: -69.35 },
    { label: "Fashion", value: -65, kept: true },
  ]);
  caption(search, "The study file stores the winner and no other trial scores: latent 32, four couplings, flow hidden 256, β = 1. The columns are that setting’s test ELBO. Fashion is higher. This is one hyperparameter set on two datasets, so the switch is on the other plots.");

  const bars = block(grid, "fvae-bars", "Train and test");
  const barBody = el("div");
  const splits = {
    digits: [
      { label: "MSE", bars: pair(0.0555, 0.0554) },
      { label: "ELBO", bars: pair(-69.4, -69.35) },
      { label: "KL", bars: pair(19.5, 19.51) },
      { label: "BPD", bars: pair(0.953, 0.953) },
      { label: "Log-det", bars: pair(-12.85, -12.87) },
    ],
    fashion: [
      { label: "MSE", bars: pair(0.0612, 0.0621) },
      { label: "ELBO", bars: pair(-64.33, -65) },
      { label: "KL", bars: pair(12.97, 12.95) },
      { label: "BPD", bars: pair(0.944, 0.945) },
      { label: "Log-det", bars: pair(-9.069, -9.072) },
    ],
  };
  switchBar(bars, "Dataset", [
    { id: "digits", label: "Digits" },
    { id: "fashion", label: "Fashion" },
  ], "digits", (id) => {
    barBody.replaceChildren();
    mountPairs(barBody, splits[id]);
  });
  bars.append(barBody);
  mountPairs(barBody, splits.digits);

  plots.append(grid);
  article.append(plots);

  discuss(article, {
    worked: linkPara([
      "Train and test agree on both full splits. Digits reach test ELBO −69.35, MSE 0.0554, and log-det −12.87. Fashion reaches ELBO −65.00, MSE 0.0621, and log-det −9.07. The grids cover many classes and are clearer than the RealNVP inverses. See the ",
      { href: "#fvae-bars", label: "bars" },
      ".",
    ]),
    unexpected: linkPara([
      "Fashion’s better ELBO comes with a worse pixel error. The shirt print is gone, and the Fashion loss is still falling at epoch 20. The pictures stay in the same soft family as the ordinary VAE. See the ",
      { href: "#fvae-gen", label: "reel" },
      ".",
    ]),
    why: linkPara([
      "Gallery samples decode N(0, I) and do not invert the flow. Bits per dimension near 0.95 are the ELBO of a Gaussian decoder. Squared error can average a logo into one code. See the ",
      { href: "#fvae-loss", label: "loss figures" },
      ".",
    ]),
    improve: linkPara([
      "Train Fashion past epoch 20. Match the ordinary VAE’s split and latent size before calling this a win. A sharper print needs a different decoder likelihood. The ",
      { href: "#fvae-optuna", label: "kept setting" },
      " is latent 32 and four couplings, and that file has no rival trials.",
    ]),
  }, [
    ["Reconstruction and likelihood", [
      el("p", null, "Reconstruction is reasonable for a Gaussian decoder: the digit and the garment class come back, and the thin parts do not. The bound is stable. Digits train and test bits per dimension are both 0.953. Fashion is 0.944 and 0.945. That is a likelihood in the form of an ELBO, and it is a different number from the RealNVP values 1.210 and 3.263. Across datasets, Fashion wins the bound and loses the pixel error."),
    ]],
    ["Against the vanilla VAE", [
      linkPara([
        "The samples are the same soft kind as the vanilla VAE, with broader class coverage than the RealNVP grid. What moved is the posterior: four couplings, and a log-det of −12.87 on digits and −9.07 on Fashion, so the flow is not an identity. Pixel error is lower than the latent-8 slice, 0.0554 against 0.0852 on digits, and that slice is a smaller experiment. Sharpness did not move. The ",
        { href: "#fvae-bars", label: "log-det bars" },
        " are the part that changed.",
      ]),
    ]],
  ]);
}

function renderBeta(article, bundle) {
  modeling(article, bundle);
  const plots = el("section", "plots");
  plots.append(el("h3", null, "Plots"));
  const grid = el("div", "plot-grid");

  const gen = block(grid, "betavae-gen", "Generation per epoch");
  mountReel(gen, frames("figures/beta", 1, 10, (epoch) => `β = 0.1 prior samples, epoch ${epoch}. Digits only.`));

  const loss = block(grid, "betavae-loss", "Loss");
  still(loss, "figures/beta/loss.png", "β-VAE training loss at β = 0.1.", "Training loss for the scored β = 0.1 run. Digits only. The per-epoch table was saved as this figure. The sweep across β is the next two charts.");

  const search = block(grid, "betavae-optuna", "Optuna");
  const trials = chartOf(bundle, "Validation loss by β").marks;
  mountStems(search, trials.map((mark) => ({
    label: String(mark.beta),
    value: mark.val_loss,
    kept: !!mark.selected,
  })));
  caption(search, "β-weighted validation loss, with every other setting held fixed. The objective is that weighted loss, so β = 0.1 is kept at 100.37. β = 1 scores 117.8, β = 4 scores 152.0, and β = 10 scores 189.5. Digits only.");

  const bars = block(grid, "betavae-bars", "Train and test");
  mountPairs(bars, [
    { label: "MSE", bars: pair(0.0874, 0.0907) },
    { label: "ELBO", bars: pair(-71.22, -73.84) },
    { label: "KL", bars: pair(26.88, 26.94) },
  ]);
  caption(bars, "The scored checkpoint, β = 0.1, on the digit slice.");

  const axes = block(grid, "betavae-axes", "Latent axes", true);
  const matrix = [0, 1, 2, 3, 4, 5, 6, 7].map((dim) => TRAVERSAL.map((betaRow) => betaRow[dim]));
  mountHeat(axes, ["0.1", "1", "4", "10"], matrix);
  caption(axes, "Each cell is how much the image changes when that coordinate moves from −3 to 3, at that β. Pale cells barely move. At β = 10 the 4th, 5th, and 6th coordinates are pale. The traversal grids themselves were not in the saved figures, so this chart is the measurement.");

  plots.append(grid);
  article.append(plots);

  discuss(article, {
    worked: linkPara([
      "Optuna’s lowest weighted loss is β = 0.1 at 100.37, with test MSE 0.0907 and all 8 axes active. Diversity peaks at β = 1, at 0.226. See the ",
      { href: "#betavae-optuna", label: "search" },
      " and the ",
      { href: "#betavae-axes", label: "axis chart" },
      ".",
    ]),
    unexpected: linkPara([
      "The search picks the reconstruction weight, and diversity is higher at β = 1. At β = 10 three axes freeze and only six stay active. See the ",
      { href: "#betavae-axes", label: "heatmap" },
      ".",
    ]),
    why: linkPara([
      "β multiplies the KL, and an axis counts as active when the variance of its posterior mean is above 0.01. MNIST has no labeled factor such as slant, so a frozen row is unused capacity. The ",
      { href: "#betavae-bars", label: "bars" },
      " are the sharp end of that trade.",
    ]),
    improve: linkPara([
      "Put diversity next to the Optuna loss, or the search keeps the small β. Keep β near 1 when the goal is the sample grid. Use a dataset with known factors if the claim is disentanglement. The ",
      { href: "#betavae-optuna", label: "loss columns" },
      " and the heatmap want different values of β.",
    ]),
  }, [
    ["What β changes", [
      el("p", null, "Raising β lifts test MSE from 0.0907 to 0.0940, 0.1114, and 0.1480. Active axes go 8, 8, 7, 6. Diversity goes 0.168, 0.226, 0.204, 0.151. Reconstruction gets worse, the code gets smaller, and the sample grid is most varied at β = 1, which is the point the weighted loss rejects. The objective explains the pick: it is the β-weighted validation loss, with hidden size, dropout, learning rate, and latent size held fixed, so a small β wins by construction."),
    ]],
    ["Moving one latent axis", [
      linkPara([
        "The ",
        { href: "#betavae-axes", label: "heatmap" },
        " is that traversal, summarized. At β = 0.1 and β = 1 every row still changes the image, about 0.04 to 0.10. At β = 10 the 4th, 5th, and 6th coordinates have KL 0.010, 0.005, and 0.004, and their responses are about 0.002, 0.001, and 0.001. Those rows barely change the picture. The rows that still move change which digit you see. That is axes turning off. It is unused capacity, and it is not a discovered factor.",
      ]),
    ]],
  ]);
}

function renderGan(article, bundle, history) {
  modeling(article, bundle);
  const plots = el("section", "plots");
  plots.append(el("h3", null, "Plots"));
  const grid = el("div", "plot-grid");

  const gen = block(grid, "gan-gen", "Generation per epoch", true);
  const genBody = el("div");
  let stopGen = () => {};
  switchBar(gen, "Dataset", [
    { id: "digits", label: "Digits" },
    { id: "fashion", label: "Fashion" },
  ], "digits", (id) => mountGen(id));
  gen.append(genBody);
  function mountGen(id) {
    stopGen();
    genBody.replaceChildren();
    if (id === "digits") {
      stopGen = mountReel(genBody, frames("figures/gan/digits", 1, 40, (epoch) => `Digits vanilla GAN, epoch ${epoch}.`));
      return;
    }
    const saved = [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39];
    stopGen = mountReel(genBody, saved.map((epoch) => ({
      epoch,
      src: `figures/gan/fashion/epoch_${String(epoch).padStart(4, "0")}.png`,
      caption: epoch === 39
        ? "Fashion-MNIST vanilla GAN, epoch 39. The scored checkpoint is epoch 40. That grid, and epochs 1, 10, 20, and 30, were not saved."
        : `Fashion-MNIST vanilla GAN, epoch ${epoch}.`,
    })));
  }
  mountGen("digits");

  const loss = block(grid, "gan-loss", "Loss", true);
  const lossBody = el("div");
  switchBar(loss, "Dataset", [
    { id: "digits", label: "Digits" },
    { id: "fashion", label: "Fashion" },
  ], "digits", (id) => showLoss(id));
  loss.append(lossBody);
  function showLoss(id) {
    lossBody.replaceChildren();
    const rows = history[id].map((row) => ({ epoch: row.epoch, d: row.eval_d, g: row.eval_g }));
    mountRibbon(lossBody, rows, [
      { key: "d", label: "D loss", color: "#1c1915" },
      { key: "g", label: "G loss", color: "var(--accent)" },
    ], 40);
    caption(lossBody, "Held-out discriminator and generator loss. The bold bar is epoch 40, the scored checkpoint.");
  }
  showLoss("digits");

  const search = block(grid, "gan-optuna", "Optuna");
  search.append(el("p", null, "The search minimizes val_gap: the absolute gap between held-out discriminator and generator loss, plus 1 when sample spread falls below 0.05. That metric favors a balanced pair of losses and penalizes a collapsed batch. The saved best_params file stores the chosen settings: noise 64, hidden 256, generator learning rate 0.0002, discriminator learning rate 0.00005, β1 0.5, batch 128. The trial table was not kept, so there is no study plot. The loss bands and the bars are the measured run."));

  const bars = block(grid, "gan-bars", "Train and test");
  const barBody = el("div");
  const splits = {
    digits: [
      { label: "D loss", bars: pair(1.401, 1.401) },
      { label: "G loss", bars: pair(0.683, 0.683) },
      { label: "Spread", bars: pair(0.0134, 0.378, "Sample", "Real") },
    ],
    fashion: [
      { label: "D loss", bars: pair(0.192, 0.204) },
      { label: "G loss", bars: pair(3.428, 3.42) },
      { label: "Spread", bars: pair(0.0413, 0.531, "Sample", "Real") },
    ],
  };
  switchBar(bars, "Dataset", [
    { id: "digits", label: "Digits" },
    { id: "fashion", label: "Fashion" },
  ], "digits", (id) => {
    barBody.replaceChildren();
    mountPairs(barBody, splits[id]);
  });
  bars.append(barBody);
  mountPairs(barBody, splits.digits);
  caption(bars, "The third pair is the scored sample spread against real images, which is the collapse check. Both score files set the collapse flag.");

  plots.append(grid);
  article.append(plots);

  discuss(article, {
    worked: linkPara([
      "Both losses are logged for 40 epochs on both datasets, and train sits on top of test. See the ",
      { href: "#gan-loss", label: "loss bands" },
      ".",
    ]),
    unexpected: linkPara([
      "Both final grids are one shape. Digits stay near discriminator 1.40 and generator 0.68, with sample spread 0.013 against 0.378 on real images. Fashion’s discriminator falls to 0.20 and the generator rises to 3.43, with spread 0.041 against 0.531. See the ",
      { href: "#gan-gen", label: "reel" },
      ".",
    ]),
    why: linkPara([
      "The loop takes one discriminator step and two generator steps, on 400 images per class, with no class label. The generator can satisfy the discriminator with a single stroke or a single shirt. The digit spread never recovers. See the ",
      { href: "#gan-bars", label: "spread bars" },
      ".",
    ]),
    improve: linkPara([
      "Condition on the class, which is the next section, and watch sample spread against real spread so a flat grid cannot hide inside a stable loss. The ",
      { href: "#gan-loss", label: "Fashion bands" },
      " are the generator climbing away from the discriminator.",
    ]),
  }, [
    ["Mode collapse in the batches", [
      linkPara([
        "The ",
        { href: "#gan-gen", label: "reel" },
        " is the check. On digits the grid is one stroke early and it is still one stroke at epoch 40. On Fashion the early frames still vary, then the batch becomes one shirt while the generator loss climbs. A healthy pair of losses would move together. These two curves do not, and the collapse flag is set on both score files. Train and test match, so the failure is the batch.",
      ]),
    ]],
  ]);
}

function renderCgan(article, bundle, history) {
  modeling(article, bundle);
  const plots = el("section", "plots");
  plots.append(el("h3", null, "Plots"));
  const grid = el("div", "plot-grid");
  const variants = [
    { id: "cgan", label: "cGAN", dir: "cgan" },
    { id: "patchgan", label: "PatchGAN", dir: "patchgan" },
    { id: "fm", label: "Feature match", dir: "fm" },
  ];
  const genState = { variant: "cgan", dataset: "digits" };
  const lossState = { variant: "cgan", dataset: "digits" };

  const gen = block(grid, "cgan-gen", "Generation per epoch", true);
  const genBody = el("div");
  let stopGen = () => {};
  switchBar(gen, "Variant", variants, genState.variant, (id) => {
    genState.variant = id;
    mountGen();
  });
  switchBar(gen, "Dataset", [
    { id: "digits", label: "Digits" },
    { id: "fashion", label: "Fashion" },
  ], genState.dataset, (id) => {
    genState.dataset = id;
    mountGen();
  });
  gen.append(genBody);
  function mountGen() {
    stopGen();
    genBody.replaceChildren();
    const variant = variants.find((item) => item.id === genState.variant);
    const list = [10, 20, 30, 40].map((epoch) => ({
      epoch,
      src: `figures/${variant.dir}/${genState.dataset}/epoch_${String(epoch).padStart(4, "0")}.png`,
      caption: `${variant.label}, ${genState.dataset}, epoch ${epoch}. Saved every 10 epochs.`,
    }));
    stopGen = mountReel(genBody, list);
  }
  mountGen();

  const loss = block(grid, "cgan-loss", "Loss", true);
  const lossBody = el("div");
  switchBar(loss, "Variant", variants, lossState.variant, (id) => {
    lossState.variant = id;
    showLoss();
  });
  switchBar(loss, "Dataset", [
    { id: "digits", label: "Digits" },
    { id: "fashion", label: "Fashion" },
  ], lossState.dataset, (id) => {
    lossState.dataset = id;
    showLoss();
  });
  loss.append(lossBody);
  function showLoss() {
    lossBody.replaceChildren();
    const key = `${lossState.variant}-${lossState.dataset}`;
    const rows = history[key].map((row) => ({
      epoch: row.epoch,
      d: row.eval_d,
      g: row.eval_g,
      bce: row.eval_g_bce,
    }));
    const series = [
      { key: "d", label: "D loss", color: "#1c1915" },
      { key: "g", label: "G loss", color: "var(--accent)" },
    ];
    if (lossState.variant === "fm") series.push({ key: "bce", label: "G BCE", color: "#8a5a2a" });
    mountRibbon(lossBody, rows, series, 40);
    caption(lossBody, "Held-out discriminator and generator loss. Feature matching adds the generator’s binary cross-entropy on its own band, because the logged generator loss includes the feature term.");
  }
  showLoss();

  const search = block(grid, "cgan-optuna", "Optuna");
  const searchBody = el("div");
  switchBar(search, "Dataset", [
    { id: "digits", label: "Digits" },
    { id: "fashion", label: "Fashion" },
  ], "digits", (id) => {
    searchBody.replaceChildren();
    mountStems(searchBody, SPREAD[id]);
  });
  search.append(searchBody);
  mountStems(searchBody, SPREAD.digits);
  caption(search, "No per-trial Optuna table was saved. The columns are sample spread divided by the spread of real images. 1 matches real images. The solid mark is the conditional model, the variant stored in best_params. Vanilla is included so the jump is visible.");

  const bars = block(grid, "cgan-bars", "Train and test");
  const barBody = el("div");
  const splits = {
    digits: [
      { label: "D loss", bars: pair(0.223, 0.216) },
      { label: "G loss", bars: pair(2.076, 2.074) },
      { label: "Spread", bars: pair(0.636, 0.607, "Sample", "Real") },
    ],
    fashion: [
      { label: "D loss", bars: pair(0.559, 0.665) },
      { label: "G loss", bars: pair(3.244, 3.257) },
      { label: "Spread", bars: pair(0.614, 0.647, "Sample", "Real") },
    ],
  };
  switchBar(bars, "Dataset", [
    { id: "digits", label: "Digits" },
    { id: "fashion", label: "Fashion" },
  ], "digits", (id) => {
    barBody.replaceChildren();
    mountPairs(barBody, splits[id]);
  });
  bars.append(barBody);
  mountPairs(barBody, splits.digits);
  caption(bars, "Train and test for the conditional model. The collapse flag is off on both. PatchGAN and feature matching are on the reel, the loss bands, and the spread chart.");

  plots.append(grid);
  article.append(plots);

  discuss(article, {
    worked: linkPara([
      "A class label lifts sample spread from a few percent of real images to about the same spread. Conditional digits are 1.05 times real spread, Fashion 0.95, and the collapse flag is off. See the ",
      { href: "#cgan-optuna", label: "spread chart" },
      ".",
    ]),
    unexpected: linkPara([
      "PatchGAN on digits still draws one broken glyph on each class row, at ratio 0.88. Feature matching pushes the generator loss to 6.54 on digits and 7.41 on Fashion, while the cross-entropy alone is 2.98 and 2.71. See the ",
      { href: "#cgan-gen", label: "reel" },
      ".",
    ]),
    why: linkPara([
      "The label tells the generator which class to aim at. A patch discriminator scores local texture, and a collapsed digit can still look sharp in a patch. Feature matching adds an L1 term of weight 10, which inflates the logged generator loss. See the ",
      { href: "#cgan-loss", label: "loss bands" },
      ".",
    ]),
    improve: linkPara([
      "Keep the conditional label. Read PatchGAN and feature matching from the class rows, and keep the feature term off the cross-entropy axis. The third band on the ",
      { href: "#cgan-loss", label: "feature-matching chart" },
      " is that split.",
    ]),
  }, [
    ["PatchGAN and feature matching", [
      linkPara([
        "Before these changes the vanilla batches are one stroke and one shirt. After the conditional model, Fashion rows separate into trousers, shirts, and shoes, noisy and different, and digit spread reaches the real-image spread. PatchGAN keeps that Fashion separation and leaves the digit rows as one broken glyph. Feature matching keeps partial digit rows, 0, 1, and 3, and speckled copies of the other classes. Diversity went up on the ",
        { href: "#cgan-optuna", label: "spread chart" },
        ". Clean, varied digits did not. The ",
        { href: "#cgan-bars", label: "conditional bars" },
        " are the run whose spread matches real images.",
      ]),
    ]],
  ]);
}

function renderPgan(article, bundle) {
  modeling(article, bundle);
  const plots = el("section", "plots");
  plots.append(el("h3", null, "Plots"));
  const grid = el("div", "plot-grid");

  const gen = block(grid, "pgan-gen", "Generation per epoch", true);
  mountReel(gen, PGAN_FILES.map(([epoch, file, phase]) => ({
    epoch,
    src: `figures/pgan/${file}`,
    caption: `Epoch ${epoch}, ${phase}. CIFAR-10.`,
  })));

  const loss = block(grid, "pgan-loss", "Loss", true);
  mountRibbon(loss, PGAN, [
    { key: "eval_d", label: "D loss", color: "#1c1915" },
    { key: "eval_g", label: "G loss", color: "var(--accent)" },
  ], 14);
  caption(loss, "Held-out discriminator and generator loss across the growth. The bold bar is epoch 14, the scored 32×32 checkpoint. Resolution changes at epochs 3, 7, and 11.");

  const search = block(grid, "pgan-optuna", "Optuna");
  mountStems(search, [
    { label: "4×4", value: 0.612 },
    { label: "8×8", value: 0.414 },
    { label: "16×16", value: 0.332 },
    { label: "32×32", value: 0.104, kept: true },
  ]);
  caption(search, "Sample standard deviation at the last epoch of each resolution. The solid mark is 32×32, the scored end of the schedule. The search minimizes val_gap, the absolute gap between the two losses plus a penalty when sample spread falls below 0.05. best_params records width 64 and learning rate 0.0002 for both networks. CIFAR-10 only.");

  const bars = block(grid, "pgan-bars", "Train and test");
  mountPairs(bars, [
    { label: "D loss", bars: pair(1.474, 1.518) },
    { label: "G loss", bars: pair(0.702, 0.703) },
    { label: "Spread", bars: pair(0.104, 0.493, "Sample", "Real") },
  ]);
  caption(bars, "The scored 32×32 checkpoint. The collapse flag stays off because 0.104 / 0.493 = 0.211, just above 0.2.");

  plots.append(grid);
  article.append(plots);

  discuss(article, {
    worked: linkPara([
      "The early 4×4 frames and the last stable 8×8 still show different fields of color, and both losses stay near 1 through the first stages. See the ",
      { href: "#pgan-gen", label: "reel" },
      " and the ",
      { href: "#pgan-loss", label: "loss bands" },
      ".",
    ]),
    unexpected: linkPara([
      "The scored 32×32 grid is one gray field. Sample standard deviation is 0.104 against 0.493 on real images. The collapse flag stays off because the ratio is 0.211. See the ",
      { href: "#pgan-bars", label: "spread bars" },
      ".",
    ]),
    why: linkPara([
      "Each new resolution gets two epochs on 2,000 images, and one of those epochs is a fade that is only half blended. That is long enough to settle on an average color. See the ",
      { href: "#pgan-optuna", label: "resolution chart" },
      ".",
    ]),
    improve: linkPara([
      "Give the stable 16×16 and 32×32 phases more epochs before judging CIFAR-10, and draw the spread ratio against 0.2 so a batch of identical gray squares cannot sit just outside the flag. The ",
      { href: "#pgan-gen", label: "reel" },
      " is the schedule as it was run.",
    ]),
  }, [
    ["Growing the resolution", [
      linkPara([
        "The schedule starts at a stable 4×4, then a fade and a stable block at 8×8, 16×16, and 32×32. During a fade the new block is mixed in by α. On this run the losses stay in a band near 1, so the optimization does not blow up as the canvas grows. The pictures do the opposite of a quality gain: sample spread falls from 0.61 at 4×4 to 0.41 at 8×8, 0.33 at 16×16, and 0.10 at 32×32. Epoch 6, the last stable 8×8, is the last batch that still looks varied. Epoch 14, the bold bar, is the scored gray field. Growing the resolution kept the loss stable and spent the sample diversity. The ",
        { href: "#pgan-optuna", label: "resolution chart" },
        " is that spend.",
      ]),
    ]],
  ]);
}

const BUILDERS = {
  realnvp: renderRealnvp,
  ae: renderAe,
  vae: renderVae,
  fvae: renderFvae,
  betavae: renderBeta,
  gan: renderGan,
  cgan: renderCgan,
  pgan: renderPgan,
};

function renderNav(bundles) {
  const toc = document.getElementById("toc");
  const jump = document.getElementById("jump");
  bundles.forEach((bundle) => {
    const item = el("li");
    const link = el("a");
    link.href = `#${bundle.id}`;
    const swatch = el("span", "swatch");
    swatch.style.setProperty("--swatch", bundle.accent);
    link.append(swatch, document.createTextNode(NAMES[bundle.id]));
    item.append(link);
    toc.append(item);
    const jumpItem = el("li");
    const jumpLink = el("a", null, NAMES[bundle.id]);
    jumpLink.href = `#${bundle.id}`;
    jumpItem.append(jumpLink);
    jump.append(jumpItem);
  });
}

function renderChapter(bundle, history) {
  const article = el("article", "section");
  article.id = bundle.id;
  article.style.setProperty("--accent", bundle.accent);
  article.append(el("h2", null, NAMES[bundle.id]));
  article.append(el("p", "question", bundle.question));
  BUILDERS[bundle.id](article, bundle, history);
  sources(article, bundle);
  document.getElementById("chapters").append(article);
}

function typeset(node) {
  const run = () => {
    if (!window.MathJax || !MathJax.typesetPromise) return;
    MathJax.typesetPromise([node]).catch((error) => console.error(error));
  };
  if (window.MathJax && MathJax.startup && MathJax.startup.promise) {
    MathJax.startup.promise.then(run);
    return;
  }
  window.setTimeout(() => typeset(node), 60);
}

async function boot() {
  if (!window.d3) {
    document.getElementById("chapters").append(el("p", null, "The charts could not be drawn."));
    return;
  }
  try {
    const manifest = await fetch("bundles/manifest.json").then((response) => response.json());
    const history = await fetch("bundles/gan-history.json").then((response) => response.json());
    const bundles = await Promise.all(manifest.map((item) => fetch(item.file).then((response) => response.json())));
    renderNav(bundles);
    bundles.forEach((bundle) => renderChapter(bundle, history));
    typeset(document.getElementById("chapters"));
    redrawCharts();
  } catch (error) {
    document.getElementById("chapters").append(el("p", null, "The report files did not load."));
    console.error(error);
  }
}

boot();
