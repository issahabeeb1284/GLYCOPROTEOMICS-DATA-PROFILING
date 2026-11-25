%% ANONYMIZED GLYCOPROTEOMICS ANALYSIS PIPELINE: 26 figures (MATLAB)
% =========================================================================
% MATLAB counterpart of python/glycoproteomics_pipeline.py. Runs on
% SIMULATED, fully anonymous intact-glycopeptide data:
%   samples ...... "Sample A" (reference) .. "Sample D"
%   proteins ..... Protein_0001 ...
%   glycosites ... Protein_0001-N123 (N-linked) / Protein_0001-T45 (O-linked)
%
% Base MATLAB only (R2020a+). No toolboxes: statistics, clustering, kernel
% density, box/violin/donut drawing are implemented in local functions.
%
% Analysis: valid-value filter -> median normalization -> left-shifted
% imputation -> Welch t-test vs Sample A -> BH FDR -> protein correction
% (glyco-delta = glycopeptide log2FC - protein log2FC) -> class / trait /
% site summaries.
%
% Output: figures/matlab/*.png (300 dpi) + ALL_FIGURES_matlab.pdf
% Random numbers differ from Python, so values differ slightly between the
% two implementations; the analysis and figure design are the same.
% =========================================================================
clear; close all; clc;
ROOT = fileparts(fileparts(mfilename('fullpath')));
OUT = fullfile(ROOT, 'figures', 'matlab');
if ~exist(OUT, 'dir'), mkdir(OUT); end
PDF = fullfile(OUT, 'ALL_FIGURES_matlab.pdf');
if exist(PDF, 'file'), delete(PDF); end
rng(7);
FDR_T = 0.05;  FC_T = 1.0;  DELTA_T = 1.0;  MIN_VALID = 0.70;

%% Theme ------------------------------------------------------------------
C.ink = hex2rgb('#0b0b0b');  C.ink2 = hex2rgb('#52514e');  C.muted = hex2rgb('#898781');
C.grid = hex2rgb('#e1e0d9'); C.ns = hex2rgb('#c3c2b7');    C.up = hex2rgb('#e34948');
C.down = hex2rgb('#2a78d6'); C.light = hex2rgb('#f0efec');
C.div = makeCmap({'#184f95','#2a78d6','#f0efec','#e34948','#8a2323'});
C.seq = makeCmap({'#b5d3f7','#6da7ec','#2a78d6','#184f95'});
SCOL = [hex2rgb('#52514e'); hex2rgb('#2a78d6'); hex2rgb('#1baf7a'); hex2rgb('#4a3aa7')];
CLASSES = {'High-mannose','Paucimannose','Hybrid','Complex','Fucosylated','Sialylated', ...
           'Sialyl+Fucosylated','O-glycan'};
CCOL = cell2mat(cellfun(@hex2rgb, {'#2a78d6','#eb6834','#1baf7a','#4a3aa7','#eda100', ...
           '#e34948','#008300','#e87ba4'}, 'UniformOutput', false)');
TCOL = [hex2rgb('#2a78d6'); hex2rgb('#e87ba4')];          % N-, O-glycosites
nC = numel(CLASSES);

%% Simulated anonymous glycoproteomics data ----------------------------------
D = simulateGlyco(350, 4, 4);
samples = D.samples;  nS = numel(samples);  nO = nS - 1;  others = samples(2:end);
grp = D.grp;  letters = cellfun(@(s) s(end), samples, 'UniformOutput', false);
short = arrayfun(@(j) sprintf('%s-R%d', letters{grp(j)}, sum(grp(1:j) == grp(j))), ...
                 1:numel(grp), 'UniformOutput', false);

[X, NORM, IMP, keep] = preprocess(D.graw, grp, MIN_VALID);
XP = preprocess(D.praw, grp, 0);
G = structfun(@(v) v(keep, :), rmfield(D.gp, 'comp'), 'UniformOutput', false);
G.comp = D.gp.comp(keep);
nG = size(X, 1);  LIN = 2.^X;
fprintf('%d glycopeptides simulated, %d retained | %d glycosites | %d N / %d O\n', ...
    size(D.graw,1), nG, numel(unique(G.site)), sum(~G.isO), sum(G.isO));

% Differential glycosylation + protein correction
[P, L2FC, ~, Q] = deAll(X, grp);
[~, PFCp] = deAll(XP, grp);
PFC = PFCp(G.prot, :);
DELTA = L2FC - PFC;
NLQ = -log10(max(Q, 1e-300));
CALL = zeros(nG, nO);  CALL(Q < FDR_T & L2FC > FC_T) = 1;  CALL(Q < FDR_T & L2FC < -FC_T) = -1;
DEC = Q < FDR_T & abs(DELTA) > DELTA_T;

classShare = shareBy(LIN, G.cls, nC, grp, nS);              % nC x nS (%)

%% 01 Glycopeptide identifications ---------------------------------------------
fig = newFig(89, 64);  ax = axes(fig);  hold(ax, 'on');
idN = sum(~isnan(D.graw(~D.gp.isO, :)), 1);  idO = sum(~isnan(D.graw(D.gp.isO, :)), 1);
b = bar(ax, [idN; idO]', 0.72, 'stacked', 'EdgeColor', 'w', 'LineWidth', 0.4);
b(1).FaceColor = TCOL(1,:);  b(2).FaceColor = TCOL(2,:);
set(ax, 'XTick', 1:numel(short), 'XTickLabel', short, 'XTickLabelRotation', 90);
ylabel(ax, 'Identified glycopeptides');  title(ax, 'Glycopeptide identifications per replicate');
legend(ax, {'N-glycopeptides', 'O-glycopeptides'}, 'Location', 'southoutside', ...
       'Orientation', 'horizontal', 'Box', 'off');
finishAxes(ax, C);  saveFig(fig, OUT, PDF, '01_glycopeptide_identifications');

%% 02 Glycan-class composition -----------------------------------------------------
fig = newFig(100, 70);  ax = axes(fig);  hold(ax, 'on');
b = bar(ax, classShare', 0.65, 'stacked', 'EdgeColor', 'w', 'LineWidth', 0.6);
for c = 1:nC, b(c).FaceColor = CCOL(c,:); end
set(ax, 'XTick', 1:nS, 'XTickLabel', samples);  ylim(ax, [0 100]);
ylabel(ax, 'Share of glycopeptide intensity (%)');  title(ax, 'Glycan-class composition');
legend(ax, fliplr(b), fliplr(CLASSES), 'Location', 'eastoutside', 'Box', 'off');
finishAxes(ax, C);  saveFig(fig, OUT, PDF, '02_glycan_class_composition');

%% 03 Glycosites per protein ----------------------------------------------------------
keptSites = unique(G.site);
nN = accumarray(D.site.prot(keptSites), ~D.site.isO(keptSites), [D.nProt 1]);
nOs = accumarray(D.site.prot(keptSites), D.site.isO(keptSites), [D.nProt 1]);
mx = max([nN; nOs]);  bins = 1:mx;
cN = histc(nN(nN > 0), bins);  cO = histc(nOs(nOs > 0), bins); %#ok<HISTC>
fig = newFig(89, 58);  ax = axes(fig);  hold(ax, 'on');
bar(ax, bins - 0.19, cN, 0.36, 'FaceColor', TCOL(1,:), 'EdgeColor', 'none');
bar(ax, bins + 0.19, cO, 0.36, 'FaceColor', TCOL(2,:), 'EdgeColor', 'none');
set(ax, 'XTick', bins);  xlabel(ax, 'Glycosites per protein');  ylabel(ax, 'Proteins');
title(ax, 'Glycosite multiplicity');
legend(ax, {'N-glycosites', 'O-glycosites'}, 'Box', 'off');
finishAxes(ax, C);  saveFig(fig, OUT, PDF, '03_glycosites_per_protein');

%% 04 Glycoforms per site --------------------------------------------------------------
perSite = accumarray(G.site, 1, [D.nSite 1]);
sIsO = D.site.isO;  has = perSite > 0;
mx = max(perSite);  bins = 1:mx;
cN = histc(perSite(has & ~sIsO), bins);  cO = histc(perSite(has & sIsO), bins); %#ok<HISTC>
fig = newFig(89, 58);  ax = axes(fig);  hold(ax, 'on');
bar(ax, bins - 0.19, cN, 0.36, 'FaceColor', TCOL(1,:), 'EdgeColor', 'none');
bar(ax, bins + 0.19, cO, 0.36, 'FaceColor', TCOL(2,:), 'EdgeColor', 'none');
xlabel(ax, 'Glycoforms detected per site');  ylabel(ax, 'Glycosites');
title(ax, sprintf('Site microheterogeneity (median %d glycoforms / site)', round(median(perSite(has)))));
legend(ax, {'N-sites', 'O-sites'}, 'Box', 'off');
finishAxes(ax, C);  saveFig(fig, OUT, PDF, '04_glycoforms_per_site');

%% 05 N-glycan composition space -----------------------------------------------------------
nm = ~G.isO;
[u, ~, ic] = unique([G.hexnac(nm) G.hex(nm) G.fuc(nm) G.neuac(nm) G.cls(nm)], 'rows');
cnt = accumarray(ic, 1);
fig = newFig(130, 75);  ax = axes(fig);  hold(ax, 'on');  grid(ax, 'on');
hs = gobjects(nC-1, 1);
for c = 1:nC-1
    m = u(:,5) == c;
    hs(c) = scatter(ax, u(m,1) + 0.13*u(m,4) - 0.15, u(m,2) + 0.22*u(m,3), cnt(m)*0.9 + 6, ...
        CCOL(c,:), 'filled', 'MarkerEdgeColor', 'w');
end
set(ax, 'XTick', 2:6, 'YTick', 3:9);  xlim(ax, [1.6 6.8]);  ylim(ax, [2.6 9.6]);
xlabel(ax, 'HexNAc (offset right = more NeuAc)');  ylabel(ax, 'Hex (offset up = more Fuc)');
title(ax, 'N-glycan composition space');
lg = legend(ax, hs, CLASSES(1:nC-1), 'Location', 'eastoutside', 'Box', 'off');
finishAxes(ax, C);  saveFig(fig, OUT, PDF, '05_nglycan_composition_space');

%% 06 Sequon usage -------------------------------------------------------------------------------
ks = keptSites(~D.site.isO(keptSites));
stc = [sum(D.site.st(ks) == 'S'), sum(D.site.st(ks) == 'T')];
[xs, ~, ix] = unique(cellstr(D.site.x(ks)));  xc = accumarray(ix, 1);  [xc, o] = sort(xc, 'descend');
fig = newFig(130, 56);  tl = tiledlayout(fig, 1, 5, 'TileSpacing', 'compact', 'Padding', 'compact');
ax = nexttile(tl);  hold(ax, 'on');
bar(ax, 1:2, stc, 0.6, 'FaceColor', C.ink2, 'EdgeColor', 'none');
for i = 1:2
    text(ax, i, stc(i), sprintf('%.0f%%', 100*stc(i)/sum(stc)), 'HorizontalAlignment', 'center', ...
         'VerticalAlignment', 'bottom', 'Color', C.ink2);
end
set(ax, 'XTick', 1:2, 'XTickLabel', {'N-X-S', 'N-X-T'}, 'XTickLabelRotation', 45);  ylabel(ax, 'N-glycosites');
title(ax, 'Sequon type');  finishAxes(ax, C);
ax = nexttile(tl, [1 4]);
bar(ax, xc, 0.65, 'FaceColor', C.ink2, 'EdgeColor', 'none');
set(ax, 'XTick', 1:numel(xc), 'XTickLabel', xs(o));  ylabel(ax, 'N-glycosites');
title(ax, 'Residue at X position (N-X-S/T; X \neq P)');
finishAxes(ax, C);  saveFig(fig, OUT, PDF, '06_sequon_usage');

%% 07 Glycosite position ---------------------------------------------------------------------------
rel = D.site.pos(keptSites) ./ D.protLen(D.site.prot(keptSites)) * 100;
isOk = D.site.isO(keptSites);
fig = newFig(89, 60);  ax = axes(fig);  hold(ax, 'on');
histogram(ax, rel(~isOk), 0:5:100, 'Normalization', 'pdf', 'DisplayStyle', 'stairs', ...
          'EdgeColor', TCOL(1,:), 'LineWidth', 1.4);
histogram(ax, rel(isOk), 0:5:100, 'Normalization', 'pdf', 'DisplayStyle', 'stairs', ...
          'EdgeColor', TCOL(2,:), 'LineWidth', 1.4);
xlabel(ax, 'Relative position in protein sequence (% from N-terminus)');  ylabel(ax, 'Density');
title(ax, 'Glycosite position along protein');
legend(ax, {'N-glycosites', 'O-glycosites'}, 'Location', 'southoutside', 'Orientation', ...
       'horizontal', 'Box', 'off');
finishAxes(ax, C);  saveFig(fig, OUT, PDF, '07_glycosite_position');

%% 08 Intensity boxplots -------------------------------------------------------------------------------
fig = newFig(130, 62);  tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
dat = {D.graw(keep,:), NORM};  ttl = {'Before normalization', 'After median normalization'};
for t = 1:2
    ax = nexttile(tl);  hold(ax, 'on');
    for j = 1:numel(grp), drawBox(ax, j, dat{t}(:,j), SCOL(grp(j),:), 0.6, 1, C); end
    set(ax, 'XTick', 1:numel(short), 'XTickLabel', short, 'XTickLabelRotation', 90);
    xlim(ax, [0.3 numel(short)+0.7]);  title(ax, ttl{t});
    if t == 1, ylabel(ax, 'log_2 glycopeptide intensity'); else, ax.YTickLabel = []; end
    finishAxes(ax, C);
end
linkaxes(tl.Children, 'y');
saveFig(fig, OUT, PDF, '08_intensity_boxplots');

%% 09 Replicate correlation --------------------------------------------------------------------------------
fig = newFig(89, 82);  ax = axes(fig);
R = corrcoef(X);
imagesc(ax, R, [floor(min(R(:))*100)/100 1]);  colormap(ax, C.seq);  axis(ax, 'square');
lbl = colorLabels(short, grp, SCOL);
set(ax, 'XTick', 1:numel(short), 'XTickLabel', lbl, 'YTick', 1:numel(short), 'YTickLabel', lbl, ...
    'XTickLabelRotation', 90, 'TickLabelInterpreter', 'tex', 'TickLength', [0 0]);
cb = colorbar(ax);  cb.Label.String = 'Pearson r';  cb.Box = 'off';
title(ax, 'Replicate correlation (glycopeptide level)');
finishAxes(ax, C);  hideAxisLines(ax, C, true, true);
saveFig(fig, OUT, PDF, '09_replicate_correlation');

%% 10 PCA --------------------------------------------------------------------------------------------------------
Xc = X' - mean(X', 1);  [U, S, ~] = svd(Xc, 'econ');  pcs = U*S;  ev = diag(S).^2;  VAR = 100*ev/sum(ev);
fig = newFig(89, 72);  ax = axes(fig);  hold(ax, 'on');  grid(ax, 'on');
hs = gobjects(nS, 1);
for s = 1:nS
    m = grp == s;
    hs(s) = scatter(ax, pcs(m,1), pcs(m,2), 40, SCOL(s,:), 'filled', 'MarkerEdgeColor', 'w');
end
yr = max(pcs(:,2)) - min(pcs(:,2));
for s = 1:nS
    m = grp == s;
    text(ax, mean(pcs(m,1)), max(pcs(m,2)) + 0.07*yr, letters{s}, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
end
padLimits(ax, 0.15);
xlabel(ax, sprintf('PC1 (%.1f%%)', VAR(1)));  ylabel(ax, sprintf('PC2 (%.1f%%)', VAR(2)));
title(ax, 'PCA of the glycoproteome');
legend(ax, hs, samples, 'Location', 'eastoutside', 'Box', 'off');
finishAxes(ax, C);  saveFig(fig, OUT, PDF, '10_pca');

%% 11 Glycopeptide volcano plots -------------------------------------------------------------------------------
xl = ceil(qtile(abs(L2FC(:)), 0.999) + 0.5);  yl = qtile(NLQ(:), 0.999) * 1.12;
fig = newFig(183, 62);  tl = tiledlayout(fig, 1, nO, 'TileSpacing', 'compact', 'Padding', 'compact');
for s = 1:nO
    ax = nexttile(tl);  hold(ax, 'on');
    scatterCalls(ax, L2FC(:,s), NLQ(:,s), CALL(:,s), C);
    xline(ax, [-FC_T FC_T], '--', 'Color', C.muted, 'LineWidth', 0.5);
    yline(ax, -log10(FDR_T), '--', 'Color', C.muted, 'LineWidth', 0.5);
    xlim(ax, [-xl xl]);  ylim(ax, [0 yl]);
    text(ax, 0.04, 0.97, sprintf('\\nabla %d', sum(CALL(:,s) == -1)), 'Units', 'normalized', ...
         'VerticalAlignment', 'top', 'Color', C.ink2);
    text(ax, 0.96, 0.97, sprintf('%d \\Delta', sum(CALL(:,s) == 1)), 'Units', 'normalized', ...
         'VerticalAlignment', 'top', 'HorizontalAlignment', 'right', 'Color', C.ink2);
    title(ax, sprintf('%s vs Sample A', others{s}));  xlabel(ax, 'Glycopeptide log_2FC');
    if s == 1, ylabel(ax, '-log_{10}(FDR)'); else, ax.YTickLabel = []; end
    finishAxes(ax, C);
end
saveFig(fig, OUT, PDF, '11_glycopeptide_volcano');

%% 12 Glycopeptide vs protein fold change -----------------------------------------------------------------------
lim = ceil(max(qtile(abs(L2FC(:)), 0.999), qtile(abs(PFC(:)), 0.999)));
fig = newFig(183, 66);  tl = tiledlayout(fig, 1, nO, 'TileSpacing', 'compact', 'Padding', 'compact');
for s = 1:nO
    ax = nexttile(tl);  hold(ax, 'on');
    b0 = ~DEC(:,s);  bu = DEC(:,s) & DELTA(:,s) > 0;  bd = DEC(:,s) & DELTA(:,s) < 0;
    scatter(ax, PFC(b0,s), L2FC(b0,s), 3, C.ns, 'filled');
    h1 = scatter(ax, PFC(bu,s), L2FC(bu,s), 6, C.up, 'filled');
    h2 = scatter(ax, PFC(bd,s), L2FC(bd,s), 6, C.down, 'filled');
    plot(ax, [-lim lim], [-lim lim], '--', 'Color', C.ink2, 'LineWidth', 0.6);
    xlim(ax, [-lim lim]);  ylim(ax, [-lim lim]);  axis(ax, 'square');
    title(ax, sprintf('%s vs Sample A', others{s}));  xlabel(ax, 'Protein log_2FC');
    if s == 1, ylabel(ax, 'Glycopeptide log_2FC'); end
    legend(ax, [h1 h2], {sprintf('Glycan-specific increase (%d)', sum(bu)), ...
           sprintf('Glycan-specific decrease (%d)', sum(bd))}, 'Location', 'southeast', ...
           'Box', 'off', 'FontSize', 5.5);
    finishAxes(ax, C);
end
title(tl, 'Glycosylation changes beyond protein abundance (off-diagonal = glycan-specific)', ...
      'FontWeight', 'bold', 'FontSize', 8);
saveFig(fig, OUT, PDF, '12_glyco_vs_protein_fold_change');

%% 13 log2FC by glycan class -------------------------------------------------------------------------------------
fig = newFig(183, 68);  tl = tiledlayout(fig, 1, nO, 'TileSpacing', 'compact', 'Padding', 'compact');
for s = 1:nO
    ax = nexttile(tl);  hold(ax, 'on');
    for c = 1:nC
        v = DELTA(G.cls == c, s);
        drawBox(ax, c, v(~isnan(v)), CCOL(c,:), 0.6, 1, C);
    end
    yline(ax, 0, 'Color', C.ink2, 'LineWidth', 0.6);
    set(ax, 'XTick', 1:nC, 'XTickLabel', CLASSES, 'XTickLabelRotation', 55);
    xlim(ax, [0.3 nC+0.7]);  title(ax, sprintf('%s vs Sample A', others{s}));
    if s == 1, ylabel(ax, 'Glyco-delta (glycopeptide - protein log_2FC)'); else, ax.YTickLabel = []; end
    finishAxes(ax, C);
end
linkaxes(tl.Children, 'y');
title(tl, 'Protein-corrected fold change by glycan class', 'FontWeight', 'bold', 'FontSize', 8);
saveFig(fig, OUT, PDF, '13_log2fc_by_glycan_class');

%% 14 Significant changes by glycan class ---------------------------------------------------------------------------
fig = newFig(100, 72);  ax = axes(fig);  hold(ax, 'on');
for s = 1:nO
    upc = accumarray(G.cls(CALL(:,s) == 1), 1, [nC 1]);
    dnc = accumarray(G.cls(CALL(:,s) == -1), 1, [nC 1]);
    bu = 0;  bd = 0;
    for c = 1:nC
        rectangle(ax, 'Position', [s-0.3, bu, 0.6, max(upc(c), eps)], 'FaceColor', CCOL(c,:), ...
                  'EdgeColor', 'w', 'LineWidth', 0.5);
        rectangle(ax, 'Position', [s-0.3, -bd-dnc(c), 0.6, max(dnc(c), eps)], 'FaceColor', CCOL(c,:), ...
                  'EdgeColor', 'w', 'LineWidth', 0.5);
        bu = bu + upc(c);  bd = bd + dnc(c);
    end
    text(ax, s, bu, num2str(bu), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'Color', C.ink2);
    text(ax, s, -bd, num2str(bd), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'Color', C.ink2);
end
yline(ax, 0, 'Color', C.ink2, 'LineWidth', 0.8);
yl = ylim(ax);  ylim(ax, yl * 1.12);  xlim(ax, [0.4 nO+0.6]);
set(ax, 'XTick', 1:nO, 'XTickLabel', others);
ax.YTickLabel = arrayfun(@(v) num2str(abs(v)), ax.YTick, 'UniformOutput', false);
ylabel(ax, {'Significant glycopeptides', '(\downarrow lower | higher \uparrow than Sample A)'});
title(ax, 'Significant changes by glycan class');
classLegend(ax, CLASSES, CCOL, 'eastoutside');
finishAxes(ax, C);  saveFig(fig, OUT, PDF, '14_significant_changes_by_class');

%% 15 Derived glycan traits ----------------------------------------------------------------------------------------------
nm = ~G.isO;  om = G.isO;
lnS = sum(LIN(nm,:), 1);  loS = sum(LIN(om,:), 1);
TR = [sum(LIN(nm & G.cls == 1, :), 1) ./ lnS * 100;
      sum(LIN(nm & G.cls == 2, :), 1) ./ lnS * 100;
      sum(LIN(nm & G.cls == 3, :), 1) ./ lnS * 100;
      sum(LIN(nm & G.fuc > 0, :), 1) ./ lnS * 100;
      sum(LIN(nm & G.neuac > 0, :), 1) ./ lnS * 100;
      sum(LIN(nm,:) .* G.neuac(nm), 1) ./ lnS;
      sum(LIN(nm & G.hexnac >= 5, :), 1) ./ lnS * 100;
      sum(LIN(om & G.hex == 0, :), 1) ./ loS * 100];
TRN = {'High-mannose (%)','Paucimannose (%)','Hybrid (%)','Fucosylation (%)','Sialylation (%)', ...
       'NeuAc per glycan','Tri/tetra-antennary (%)','Truncated O-glycans (%)'};
show = [1 4 5 6 7 8];
fig = newFig(183, 88);  tl = tiledlayout(fig, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
for k = show
    ax = nexttile(tl);  hold(ax, 'on');
    for s = 1:nS
        v = TR(k, grp == s);
        bar(ax, s, mean(v), 0.62, 'FaceColor', SCOL(s,:), 'EdgeColor', 'none');
        errorbar(ax, s, mean(v), std(v), 'Color', C.ink, 'LineWidth', 0.8, 'CapSize', 3);
        scatter(ax, s + (rand(size(v)) - 0.5)*0.26, v, 12, 'w', 'filled', 'MarkerEdgeColor', C.ink);
    end
    lo = min(TR(k,:));  hi = max(TR(k,:));
    ylim(ax, [max(0, lo - (hi-lo)*0.8), hi + (hi-lo)*0.4]);
    set(ax, 'XTick', 1:nS, 'XTickLabel', letters);  title(ax, TRN{k}, 'FontSize', 7);
    finishAxes(ax, C);
end
title(tl, 'Derived glycan traits per sample (bars = mean \pm SD, dots = replicates)', ...
      'FontWeight', 'bold', 'FontSize', 8);
saveFig(fig, OUT, PDF, '15_derived_glycan_traits');

%% 16 Derived-trait heatmap ----------------------------------------------------------------------------------------------------
tm = zeros(numel(TRN), nS);
for s = 1:nS, tm(:,s) = mean(TR(:, grp == s), 2); end
tfc = log2(tm(:, 2:end) ./ tm(:,1));
tp = zeros(size(tfc));
for s = 1:nO, tp(:,s) = welchT(TR(:, grp == s+1), TR(:, grp == 1)); end
fig = newFig(100, 72);  ax = axes(fig);
lim = max(abs(tfc(:)));
imagesc(ax, tfc, [-lim lim]);  colormap(ax, C.div);
for i = 1:size(tfc,1)
    for j = 1:nO
        if tp(i,j) < 0.05, st = '*'; else, st = ''; end
        if abs(tfc(i,j)) > 0.6*lim, tc = 'w'; else, tc = C.ink; end
        text(ax, j, i, sprintf('%+.2f%s', tfc(i,j), st), 'HorizontalAlignment', 'center', ...
             'FontSize', 6.5, 'Color', tc);
    end
end
set(ax, 'XTick', 1:nO, 'XTickLabel', others, 'YTick', 1:numel(TRN), 'YTickLabel', TRN, 'TickLength', [0 0]);
cb = colorbar(ax);  cb.Label.String = 'log_2 ratio vs Sample A';  cb.Box = 'off';
title(ax, 'Derived-trait remodeling (* p < 0.05)');
finishAxes(ax, C);  hideAxisLines(ax, C, true, true);
saveFig(fig, OUT, PDF, '16_derived_trait_heatmap');

%% 17 Clustered heatmap of top glycopeptides -----------------------------------------------------------------------------------
top = [];
for s = 1:nO
    idx = find(CALL(:,s) ~= 0);  [~, o] = sort(Q(idx,s));  top = [top; idx(o(1:min(20, numel(o))))]; %#ok<AGROW>
end
top = unique(top, 'stable');
Zm = X(top,:);  Zm = (Zm - mean(Zm,2)) ./ std(Zm, 0, 2);
[~, ro] = upgma(1 - corrcoef(Zm'));  Zm = Zm(ro,:);  topc = G.cls(top(ro));
fig = newFig(89, 112);
axc = axes(fig, 'Position', [0.10 0.20 0.03 0.72]);
image(axc, reshape(CCOL(topc,:), [], 1, 3));  axis(axc, 'off');
ax = axes(fig, 'Position', [0.15 0.20 0.62 0.72]);
imagesc(ax, Zm, [-2.5 2.5]);  colormap(ax, C.div);  hold(ax, 'on');
bounds = cumsum(arrayfun(@(s) sum(grp == s), 1:nS));
for bd = bounds(1:end-1), xline(ax, bd + 0.5, 'Color', 'w', 'LineWidth', 1.5); end
set(ax, 'XTick', 1:numel(short), 'XTickLabel', colorLabels(short, grp, SCOL), 'XTickLabelRotation', 90, ...
    'YTick', [], 'TickLabelInterpreter', 'tex', 'TickLength', [0 0]);
cb = colorbar(ax, 'Position', [0.80 0.55 0.025 0.25]);  cb.Label.String = 'Row z-score';  cb.Box = 'off';
title(ax, sprintf('Top %d differentially abundant glycopeptides', numel(top)));
finishAxes(ax, C);  hideAxisLines(ax, C, true, true);
present = unique(topc)';
hl = gobjects(numel(present), 1);
for i = 1:numel(present), hl(i) = patch(ax, NaN, NaN, CCOL(present(i),:), 'EdgeColor', 'none'); end
legend(ax, hl, CLASSES(present), 'Position', [0.15 0.01 0.62 0.08], 'NumColumns', 3, 'Box', 'off', ...
       'FontSize', 6);
saveFig(fig, OUT, PDF, '17_top_glycopeptide_heatmap');

%% 18 Site-specific glycoform profiles ---------------------------------------------------------------------------------------------
siteN = accumarray(G.site, 1, [D.nSite 1]);
score = accumarray(G.site, sum(abs(DELTA), 2, 'omitnan'), [D.nSite 1]);
score(siteN < 6 | D.site.isO) = -Inf;
[~, o] = sort(score, 'descend');  topSites = o(1:3);
fig = newFig(183, 74);  tl = tiledlayout(fig, 1, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
for k = 1:3
    ax = nexttile(tl);  hold(ax, 'on');
    gi = find(G.site == topSites(k));
    [~, oo] = sortrows([G.cls(gi) G.hexnac(gi) G.hex(gi)]);  gi = gi(oo);
    rel = zeros(numel(gi), nS);
    for s = 1:nS, rel(:,s) = mean(LIN(gi, grp == s), 2); end
    rel = rel ./ sum(rel, 1) * 100;
    for s = 1:nS
        b0 = 0;
        for g = 1:numel(gi)
            rectangle(ax, 'Position', [s-0.35, b0, 0.7, rel(g,s)], 'FaceColor', CCOL(G.cls(gi(g)),:), ...
                      'EdgeColor', 'w', 'LineWidth', 0.6);
            b0 = b0 + rel(g,s);
        end
    end
    xlim(ax, [0.4 nS+0.6]);  ylim(ax, [0 100]);
    set(ax, 'XTick', 1:nS, 'XTickLabel', letters);
    title(ax, D.site.name{topSites(k)}, 'FontSize', 7, 'Interpreter', 'none');
    if k == 1, ylabel(ax, 'Relative glycoform abundance at site (%)'); else, ax.YTickLabel = []; end
    finishAxes(ax, C);
end
ax = nexttile(tl);  axis(ax, 'off');
present = unique(G.cls(ismember(G.site, topSites)))';
hl = gobjects(numel(present), 1);
for i = 1:numel(present), hl(i) = patch(ax, NaN, NaN, CCOL(present(i),:), 'EdgeColor', 'none'); end
legend(ax, hl, CLASSES(present), 'Location', 'west', 'Box', 'off');
title(tl, 'Site-specific glycoform profiles (each segment = one glycoform)', 'FontWeight', 'bold', 'FontSize', 8);
saveFig(fig, OUT, PDF, '18_site_glycoform_profiles');

%% 19 UpSet overlap -------------------------------------------------------------------------------------------------------------------
sigM = CALL ~= 0;  combos = [];  cnt = [];
for mask = 1:(2^nO - 1)
    inS = bitget(mask, 1:nO) == 1;
    ex = all(sigM(:, inS), 2) & ~any(sigM(:, ~inS), 2);
    if any(ex), combos = [combos; inS]; cnt = [cnt; sum(ex)]; end %#ok<AGROW>
end
[cnt, o] = sort(cnt, 'descend');  combos = combos(o,:);  nc = numel(cnt);
fig = newFig(89, 74);  tl = tiledlayout(fig, 3, 1, 'TileSpacing', 'none', 'Padding', 'compact');
ax = nexttile(tl, [2 1]);  hold(ax, 'on');
bar(ax, 1:nc, cnt, 0.6, 'FaceColor', C.ink2, 'EdgeColor', 'none');
for i = 1:nc
    text(ax, i, cnt(i), num2str(cnt(i)), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
         'FontSize', 6, 'Color', C.ink2);
end
xlim(ax, [0.4 nc+0.6]);  ylim(ax, [0 max(cnt)*1.15]);  ylabel(ax, 'Glycopeptides');
title(ax, 'Shared and unique glycopeptide changes');  finishAxes(ax, C);  ax.XAxis.Visible = 'off';
ax = nexttile(tl);  hold(ax, 'on');
for i = 1:nc
    yy = nO + 1 - find(combos(i,:));
    if numel(yy) > 1, plot(ax, [i i], [min(yy) max(yy)], '-', 'Color', C.ink2, 'LineWidth', 1); end
    for s = 1:nO
        if combos(i,s), cc = SCOL(s+1,:); sz = 26; else, cc = C.grid; sz = 16; end
        scatter(ax, i, nO + 1 - s, sz, cc, 'filled');
    end
end
xlim(ax, [0.4 nc+0.6]);  ylim(ax, [0.4 nO+0.6]);
set(ax, 'YTick', 1:nO, 'YTickLabel', fliplr(others), 'TickLength', [0 0]);
finishAxes(ax, C);  ax.XAxis.Visible = 'off';  hideAxisLines(ax, C, false, true);
saveFig(fig, OUT, PDF, '19_upset_changed_glycopeptides');

%% 20 Ranked glyco-delta waterfall ---------------------------------------------------------------------------------------------------------
fig = newFig(89, 88);  tl = tiledlayout(fig, nO, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
for s = 1:nO
    ax = nexttile(tl);  hold(ax, 'on');
    ok = ~isnan(DELTA(:,s));  [v, o] = sort(DELTA(ok,s), 'descend');  d = DEC(ok,s);  d = d(o);
    cd = repmat(C.ns, numel(v), 1);  cd(d & v > 0, :) = repmat(C.up, sum(d & v > 0), 1);
    cd(d & v < 0, :) = repmat(C.down, sum(d & v < 0), 1);
    b = bar(ax, 1:numel(v), v, 1, 'FaceColor', 'flat', 'EdgeColor', 'none');  b.CData = cd;
    yline(ax, 0, 'Color', C.ink2, 'LineWidth', 0.5);  xlim(ax, [0 numel(v)+1]);
    text(ax, 0.99, 0.92, others{s}, 'Units', 'normalized', 'HorizontalAlignment', 'right', ...
         'VerticalAlignment', 'top', 'FontWeight', 'bold');
    ylabel(ax, 'Glyco-delta');
    if s == 1, title(ax, 'Ranked glycan-specific changes (colored = significant & |\Delta| > 1)'); end
    if s < nO, ax.XTickLabel = []; else, xlabel(ax, 'Glycopeptides ranked by protein-corrected change'); end
    finishAxes(ax, C);
end
linkaxes(tl.Children, 'y');
saveFig(fig, OUT, PDF, '20_ranked_glyco_delta_waterfall');

%% 21 Composition shift maps (core x sialylation) ---------------------------------------------------------------------------------------------
nm = find(~G.isO);
coreKey = [G.hexnac(nm) G.hex(nm) double(G.fuc(nm) > 0)];
[cores, ~, ci] = unique(coreKey, 'rows');
coreLab = arrayfun(@(i) sprintf('HexNAc%dHex%d%s', cores(i,1), cores(i,2), repmat('+Fuc', 1, cores(i,3))), ...
                   (1:size(cores,1))', 'UniformOutput', false);
fig = newFig(183, 95);  tl = tiledlayout(fig, 1, nO, 'TileSpacing', 'compact', 'Padding', 'compact');
for s = 1:nO
    ax = nexttile(tl);
    gridV = accumarray([ci, G.neuac(nm) + 1], DELTA(nm,s), [size(cores,1) 5], @(v) mean(v, 'omitnan'), NaN);
    h = imagesc(ax, gridV, [-1.5 1.5]);  set(h, 'AlphaData', ~isnan(gridV));  colormap(ax, C.div);
    set(ax, 'XTick', 1:5, 'XTickLabel', 0:4, 'YTick', 1:size(cores,1), 'TickLength', [0 0]);
    if s == 1
        ax.YTickLabel = coreLab;  ax.YAxis.FontSize = 5.5;  ylabel(ax, 'Glycan core (HexNAc/Hex/Fuc)');
    else
        ax.YTickLabel = [];
    end
    xlabel(ax, 'NeuAc residues');  title(ax, sprintf('%s vs Sample A', others{s}));
    finishAxes(ax, C);  hideAxisLines(ax, C, true, true);
end
cb = colorbar(ax);  cb.Layout.Tile = 'east';  cb.Label.String = 'Mean glyco-delta';  cb.Box = 'off';
title(tl, 'N-glycan composition shift maps: core \times sialylation (blank = not observed)', ...
      'FontWeight', 'bold', 'FontSize', 8);
saveFig(fig, OUT, PDF, '21_composition_shift_maps');

%% 22 Glycan-class donuts ------------------------------------------------------------------------------------------------------------------------
fig = newFig(183, 62);  tl = tiledlayout(fig, 1, nS, 'TileSpacing', 'compact', 'Padding', 'compact');
for s = 1:nS
    ax = nexttile(tl);  hold(ax, 'on');
    drawDonut(ax, classShare(:,s), CCOL, 0.62);
    text(ax, 0, 0, letters{s}, 'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold', ...
         'Color', SCOL(s,:));
    axis(ax, 'equal');  axis(ax, 'off');  title(ax, samples{s}, 'FontSize', 7);
end
hl = gobjects(nC, 1);
for c = 1:nC, hl(c) = patch(ax, NaN, NaN, CCOL(c,:), 'EdgeColor', 'none'); end
lg = legend(ax, hl, CLASSES, 'NumColumns', 4, 'Box', 'off');  lg.Layout.Tile = 'south';
title(tl, 'Glycan-class share of total glycopeptide intensity', 'FontWeight', 'bold', 'FontSize', 8);
saveFig(fig, OUT, PDF, '22_glycan_class_donuts');

%% 23 Site microheterogeneity (entropy) -------------------------------------------------------------------------------------------------------------
multi = find(siteN >= 3);  inM = ismember(G.site, multi);
ENT = zeros(numel(multi), nS);
[~, sidx] = ismember(G.site(inM), multi);
for s = 1:nS
    m = mean(LIN(inM, grp == s), 2);
    tot = accumarray(sidx, m);  p = m ./ tot(sidx);
    ENT(:,s) = accumarray(sidx, -p .* log2(p));
end
fig = newFig(89, 64);  ax = axes(fig);  hold(ax, 'on');
for s = 1:nS
    drawViolin(ax, s, ENT(:,s), SCOL(s,:), 0.4);
    md = median(ENT(:,s));  plot(ax, s + [-0.18 0.18], [md md], 'w', 'LineWidth', 1.6);
    if s > 1
        text(ax, s, max(ENT(:,s)) + 0.15, sprintf('p = %.1e', signedRankP(ENT(:,s) - ENT(:,1))), ...
             'HorizontalAlignment', 'center', 'FontSize', 6, 'Color', C.ink2);
    end
end
set(ax, 'XTick', 1:nS, 'XTickLabel', samples);  xlim(ax, [0.4 nS+0.6]);  ylim(ax, [0 max(ENT(:)) + 0.7]);
ylabel(ax, {'Glycoform diversity per site', '(Shannon entropy, bits)'});
title(ax, sprintf('Site microheterogeneity (%d sites with \\geq3 glycoforms)', numel(multi)));
finishAxes(ax, C);  saveFig(fig, OUT, PDF, '23_site_microheterogeneity_entropy');

%% 24 Sialic-acid distribution -------------------------------------------------------------------------------------------------------------------------
sa = shareBy(LIN(~G.isO,:), G.neuac(~G.isO) + 1, 5, grp, nS);
fig = newFig(89, 60);  ax = axes(fig);
b = bar(ax, 0:4, sa, 'grouped', 'EdgeColor', 'none');
for s = 1:nS, b(s).FaceColor = SCOL(s,:); end
xlabel(ax, 'NeuAc residues per N-glycan');  ylabel(ax, 'Share of N-glycopeptide intensity (%)');
title(ax, 'Sialic-acid content distribution');  legend(ax, samples, 'Box', 'off');
finishAxes(ax, C);  saveFig(fig, OUT, PDF, '24_sialic_acid_distribution');

%% 25 Top changed glycosites ------------------------------------------------------------------------------------------------------------------------------
present = unique(G.site);
fig = newFig(183, 74);  tl = tiledlayout(fig, 1, nO, 'TileSpacing', 'loose', 'Padding', 'compact');
refSum = accumarray(G.site, mean(LIN(:, grp == 1), 2), [D.nSite 1]);
for s = 1:nO
    num = accumarray(G.site, mean(LIN(:, grp == s+1), 2), [D.nSite 1]);
    sfc = log2(num(present) ./ refSum(present)) - PFCp(D.site.prot(present), s);
    [~, o] = sort(abs(sfc), 'descend');  o = o(1:12);  [v, oo] = sort(sfc(o));  names = D.site.name(present(o(oo)));
    ax = nexttile(tl);  hold(ax, 'on');
    for i = 1:numel(v)
        if v(i) > 0, cc = C.up; else, cc = C.down; end
        plot(ax, [0 v(i)], [i i], 'Color', cc, 'LineWidth', 1.2);
        scatter(ax, v(i), i, 22, cc, 'filled', 'MarkerEdgeColor', 'w');
    end
    xline(ax, 0, 'Color', C.ink2, 'LineWidth', 0.6);
    set(ax, 'YTick', 1:numel(v), 'YTickLabel', names, 'TickLabelInterpreter', 'none');
    ax.YAxis.FontSize = 5.5;  ylim(ax, [0.3 numel(v)+0.7]);
    xlabel(ax, 'Site-level log_2FC (protein-corrected)');  title(ax, sprintf('%s vs Sample A', others{s}));
    finishAxes(ax, C);
end
title(tl, 'Most remodeled glycosites', 'FontWeight', 'bold', 'FontSize', 8);
saveFig(fig, OUT, PDF, '25_top_changed_glycosites');

%% 26 Glycoprotein site maps ----------------------------------------------------------------------------------------------------------------------------------
sm = 2;                                                  % Sample C
nsig = accumarray(G.prot, DEC(:,sm), [D.nProt 1]);
[~, o] = sort(nsig, 'descend');  prots = o(1:6);
sd = accumarray(G.site, DELTA(:,sm), [D.nSite 1], @(v) mean(v, 'omitnan'), NaN);
maxlen = max(D.protLen(prots));
fig = newFig(183, 82);  ax = axes(fig);  hold(ax, 'on');
cmap = C.div;  cval = @(v) cmap(max(1, min(256, round((v + 3)/6*255) + 1)), :);
for i = 1:numel(prots)
    y = numel(prots) - i;  p = prots(i);
    rectangle(ax, 'Position', [0, y - 0.09, D.protLen(p), 0.18], 'Curvature', [0.02 1], ...
              'FaceColor', C.grid, 'EdgeColor', 'none');
    ps = find(D.site.prot == p & ~isnan(sd));
    [~, oo] = sort(D.site.pos(ps));  ps = ps(oo);  lastLab = -Inf;
    for k = ps'
        if D.site.isO(k), stem = 0.26; mk = 's'; ms = 30; else, stem = 0.38; mk = 'o'; ms = 46; end
        plot(ax, [D.site.pos(k) D.site.pos(k)], [y + 0.09, y + stem], 'Color', C.ink2, 'LineWidth', 0.7);
        scatter(ax, D.site.pos(k), y + stem, ms, cval(sd(k)), 'filled', 'Marker', mk, 'MarkerEdgeColor', C.ink2);
        if D.site.pos(k) - lastLab > maxlen*0.045
            nm_ = strsplit(D.site.name{k}, '-');
            text(ax, D.site.pos(k), y + stem + 0.1, nm_{2}, 'HorizontalAlignment', 'center', ...
                 'VerticalAlignment', 'bottom', 'FontSize', 5, 'Color', C.ink2);
            lastLab = D.site.pos(k);
        end
    end
    text(ax, -maxlen*0.01, y, sprintf('Protein\\_%04d', p), 'HorizontalAlignment', 'right', 'FontSize', 6.5);
end
xlim(ax, [-maxlen*0.13, maxlen*1.02]);  ylim(ax, [-0.5, numel(prots) - 0.1 + 0.5]);
ax.YAxis.Visible = 'off';  xlabel(ax, 'Residue position');
colormap(ax, C.div);  caxis(ax, [-3 3]);
cb = colorbar(ax);  cb.Label.String = 'Mean glyco-delta at site';  cb.Box = 'off';
h1 = scatter(ax, NaN, NaN, 40, 'w', 'o', 'MarkerEdgeColor', C.ink2);
h2 = scatter(ax, NaN, NaN, 30, 'w', 's', 'MarkerEdgeColor', C.ink2);
legend(ax, [h1 h2], {'N-glycosite', 'O-glycosite'}, 'Location', 'southeast', 'Orientation', ...
       'horizontal', 'Box', 'off');
title(ax, sprintf('Glycoprotein site maps: %s vs Sample A (proteins with most glycan-specific changes)', others{sm}));
finishAxes(ax, C);  saveFig(fig, OUT, PDF, '26_glycoprotein_site_maps');

fprintf('26 figures saved to %s\n', OUT);

%% ============================ LOCAL FUNCTIONS =================================
function D = simulateGlyco(nProt, nS, nR)
    AA = 'ACDEFGHIKLMNQRSTVWY';
    L = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
    D.samples = arrayfun(@(i) sprintf('Sample %s', L(i)), 1:nS, 'UniformOutput', false);
    D.nProt = nProt;
    [nLib, oLib] = glycanLibrary();
    D.protLen = randi([250 1599], nProt, 1);
    pBase = 24 + 1.8*randn(nProt, 1);
    pEff = zeros(nProt, nS);
    for s = 2:nS
        hit = rand(nProt, 1) < 0.10;
        pEff(hit, s) = (2*(rand(sum(hit),1) < 0.5) - 1) .* (1.3 + 0.4*randn(sum(hit),1));
    end
    gp = struct('prot', [], 'site', [], 'isO', [], 'hexnac', [], 'hex', [], 'fuc', [], 'neuac', [], ...
                'cls', [], 'comp', {{}});
    st = struct('prot', [], 'isO', [], 'pos', [], 'x', '', 'st', '', 'name', {{}});
    base = [];  nSite = 0;
    for p = 1:nProt
        nN = min(geornd1(0.45), 8);
        if rand < 0.30, nOs = randi([1 3]); else, nOs = 0; end
        pos = sort(20 + randperm(D.protLen(p) - 40, nN + nOs));
        kinds = [false(1, nN) true(1, nOs)];  kinds = kinds(randperm(numel(kinds)));
        for k = 1:numel(pos)
            nSite = nSite + 1;  isO = kinds(k);
            if ~isO
                xr = AA(randi(numel(AA)));  if rand < 0.4, s_t = 'S'; else, s_t = 'T'; end
                lab = sprintf('N%d', pos(k));
                m = betarnd1(1.2, 2.2);  sp = betarnd1(2, 2);  fp = betarnd1(1.5, 1.5);
                lib = nLib;  w = zeros(size(lib,1), 1);
                for g = 1:size(lib,1)
                    c = classOf(lib(g,:), false);
                    if c == 1, w(g) = 3*m;
                    elseif c == 2, w(g) = 0.25;
                    elseif c == 3, w(g) = 0.5;
                    else
                        w(g) = (1-m) * pick(lib(g,4) > 0, sp, 1-sp) * 2 * pick(lib(g,3) > 0, fp, 1-fp);
                    end
                end
                w = w + 1e-3;  kk = randi([2 13]);
            else
                xr = ' ';  s_t = ' ';
                if rand < 0.4, res = 'S'; else, res = 'T'; end
                lab = sprintf('%s%d', res, pos(k));
                lib = oLib;  w = [0.5 0.6 1 1 0.8 0.6 0.6 0.4 0.3]';  kk = randi([1 5]);
            end
            st.prot(nSite,1) = p;  st.isO(nSite,1) = isO;  st.pos(nSite,1) = pos(k);
            st.x(nSite,1) = xr;  st.st(nSite,1) = s_t;  st.name{nSite,1} = sprintf('Protein_%04d-%s', p, lab);
            chosen = weightedSample(w / sum(w), min(kk, size(lib,1)));
            share = dirichletRnd(0.8, numel(chosen));
            for j = 1:numel(chosen)
                g = lib(chosen(j), :);
                gp.prot(end+1,1) = p;  gp.site(end+1,1) = nSite;  gp.isO(end+1,1) = isO;
                gp.hexnac(end+1,1) = g(1);  gp.hex(end+1,1) = g(2);  gp.fuc(end+1,1) = g(3);
                gp.neuac(end+1,1) = g(4);  gp.cls(end+1,1) = classOf(g, isO);
                gp.comp{end+1,1} = compName(g);
                base(end+1,1) = pBase(p) + 0.9*log2(share(j)*numel(chosen)) - 1 + 0.7*randn; %#ok<AGROW>
            end
        end
    end
    gp.isO = logical(gp.isO);  st.isO = logical(st.isO);
    n = numel(base);  eff = zeros(n, nS);
    mult = @() (1 + 0.3*randn(n,1)) .* (rand(n,1) < 0.75);
    eff(:,2) = 1.3 * (gp.fuc > 0 & ~gp.isO) .* mult();
    eff(:,3) = (1.4*(gp.neuac >= 2) - 0.7*(gp.cls == 4) + 0.6*(gp.hexnac >= 5)) .* mult();
    eff(:,4) = (-1.5*(gp.cls == 1) + 1.8*(gp.isO & gp.hex == 0)) .* mult();
    for s = 2:nS
        r = rand(n,1) < 0.02;
        eff(r,s) = eff(r,s) + (2*(rand(sum(r),1) < 0.5) - 1) .* (1.5 + 0.3*randn(sum(r),1));
    end
    graw = zeros(n, nS*nR);  praw = zeros(nProt, nS*nR);  grp = zeros(1, nS*nR);  c = 0;
    for s = 1:nS
        for r = 1:nR
            c = c + 1;  off = 0.2*randn;  grp(c) = s;
            graw(:,c) = base + eff(:,s) + pEff(gp.prot, s) + 0.4*randn(n,1) + off;
            praw(:,c) = pBase + pEff(:,s) + 0.25*randn(nProt,1) + off;
        end
    end
    low = graw < qtile(graw(:), 0.08);
    graw((low & rand(size(graw)) < 0.6) | rand(size(graw)) < 0.02) = NaN;
    D.gp = gp;  D.site = st;  D.nSite = nSite;  D.graw = graw;  D.praw = praw;  D.grp = grp;
end

function [nLib, oLib] = glycanLibrary()
    nLib = [repmat(2, 5, 1) (5:9)' zeros(5, 2)];
    nLib = [nLib; 2 3 0 0; 2 3 1 0; 2 4 0 0; 3 6 0 0; 3 5 0 0; 3 6 1 0; 3 6 0 1];
    cores = [3 4; 4 4; 4 5; 5 6; 6 7];
    for i = 1:size(cores, 1)
        hn = cores(i,1);  maxF = 1 + (hn >= 5);
        for f = 0:maxF
            for s = 0:hn-2, nLib = [nLib; hn cores(i,2) f s]; end %#ok<AGROW>
        end
    end
    oLib = [1 0 0 0; 1 0 0 1; 1 1 0 0; 1 1 0 1; 1 1 0 2; 2 2 0 0; 2 2 0 1; 2 2 0 2; 2 2 1 1];
end

function c = classOf(g, isO)
    hn = g(1);  hx = g(2);  f = g(3);  s = g(4);
    if isO, c = 8;
    elseif hn == 2 && hx >= 5, c = 1;
    elseif hn == 2, c = 2;
    elseif hn == 3 && hx >= 5, c = 3;
    elseif f && s, c = 7;
    elseif s, c = 6;
    elseif f, c = 5;
    else, c = 4;
    end
end

function n = compName(g)
    lab = {'HexNAc', 'Hex', 'Fuc', 'NeuAc'};  n = '';
    for i = 1:4, if g(i), n = [n sprintf('%s(%d)', lab{i}, g(i))]; end, end %#ok<AGROW>
end

function v = pick(cond, a, b), if cond, v = a; else, v = b; end, end

function k = geornd1(p), k = 1; while rand > p, k = k + 1; end, end

function x = gammaRnd(a)
    % Marsaglia-Tsang gamma sampler (a > 0)
    if a < 1, x = gammaRnd(a + 1) * rand^(1/a); return; end
    d = a - 1/3;  c = 1/sqrt(9*d);
    while true
        z = randn;  v = (1 + c*z)^3;
        if v > 0 && log(rand) < 0.5*z^2 + d - d*v + d*log(v), x = d*v; return; end
    end
end

function x = betarnd1(a, b), ga = gammaRnd(a);  x = ga / (ga + gammaRnd(b)); end

function w = dirichletRnd(a, k), w = arrayfun(@(i) gammaRnd(a), 1:k);  w = w / sum(w); end

function idx = weightedSample(p, k)
    idx = zeros(1, k);  p = p(:)';
    for i = 1:k
        j = find(cumsum(p) / sum(p) >= rand, 1);  idx(i) = j;  p(j) = 0;
    end
end

function [X, NORM, IMP, keep] = preprocess(raw, grp, minValid)
    nS = max(grp);  vf = zeros(size(raw,1), nS);
    for s = 1:nS, vf(:,s) = mean(~isnan(raw(:, grp == s)), 2); end
    keep = any(vf >= minValid, 2);
    r = raw(keep,:);  med = median(r, 1, 'omitnan');
    NORM = r - med + mean(med);  IMP = isnan(NORM);  X = NORM;
    for j = 1:size(X,2)
        mu = mean(NORM(:,j), 'omitnan');  sd = std(NORM(:,j), 'omitnan');
        X(IMP(:,j), j) = (mu - 1.8*sd) + 0.3*sd*randn(sum(IMP(:,j)), 1);
    end
end

function [P, FC, T, Q] = deAll(X, grp)
    nS = max(grp);  ref = X(:, grp == 1);
    P = zeros(size(X,1), nS-1);  FC = P;  T = P;  Q = P;
    for s = 2:nS
        [P(:,s-1), FC(:,s-1), T(:,s-1)] = welchT(X(:, grp == s), ref);
        Q(:,s-1) = bhFDR(P(:,s-1));
    end
end

function [p, d, t] = welchT(a, b)
    n1 = size(a,2);  n2 = size(b,2);  v1 = var(a, 0, 2);  v2 = var(b, 0, 2);
    d = mean(a,2) - mean(b,2);  se2 = v1/n1 + v2/n2;  t = d ./ sqrt(se2);
    df = se2.^2 ./ ((v1/n1).^2/(n1-1) + (v2/n2).^2/(n2-1));
    p = betainc(df ./ (df + t.^2), df/2, 0.5);
end

function q = bhFDR(p)
    p = p(:);  n = numel(p);  [ps, o] = sort(p);
    qs = flipud(cummin(flipud(ps .* n ./ (1:n)')));  q = zeros(n,1);  q(o) = min(qs, 1);
end

function p = signedRankP(d)
    % Wilcoxon signed-rank test, normal approximation (two-sided)
    d = d(d ~= 0 & ~isnan(d));  n = numel(d);
    [~, o] = sort(abs(d));  r = zeros(n,1);  r(o) = 1:n;
    [ua, ~, ia] = unique(abs(d));                                   % average ties
    if numel(ua) < n, r = accumarray(ia, r, [], @mean);  r = r(ia); end
    W = sum(r(d > 0));  mu = n*(n+1)/4;  sg = sqrt(n*(n+1)*(2*n+1)/24);
    p = erfc(abs(W - mu) / sg / sqrt(2));
end

function S = shareBy(LIN, key, nK, grp, nS)
    nR = size(LIN, 2);  per = zeros(nK, nR);
    for j = 1:nR, per(:,j) = accumarray(key, LIN(:,j), [nK 1]) / sum(LIN(:,j)) * 100; end
    S = zeros(nK, nS);
    for s = 1:nS, S(:,s) = mean(per(:, grp == s), 2); end
end

function v = qtile(x, q)
    x = sort(x(~isnan(x(:))));  pos = 1 + (numel(x)-1)*q;  lo = floor(pos);  hi = ceil(pos);
    v = x(lo) + (pos - lo)*(x(hi) - x(lo));
end

function f = kde(x, g)
    x = x(:);  h = 1.06*std(x)*numel(x)^(-1/5);  f = zeros(size(g));
    for i = 1:numel(g), f(i) = mean(exp(-0.5*((g(i) - x)/h).^2)); end
    f = f / (h*sqrt(2*pi));
end

function [Z, order] = upgma(D)
    n = size(D,1);  D(1:n+1:end) = Inf;  active = true(1,n);  sz = ones(1,n);
    ids = 1:n;  members = num2cell(1:n);  Z = zeros(n-1, 3);
    for k = 1:n-1
        Dm = D;  Dm(~active,:) = Inf;  Dm(:,~active) = Inf;
        [m, idx] = min(Dm(:));  [i, j] = ind2sub([n n], idx);  Z(k,:) = [ids(i) ids(j) m];
        nr = (sz(i)*D(i,:) + sz(j)*D(j,:)) / (sz(i) + sz(j));
        D(i,:) = nr;  D(:,i) = nr';  D(i,i) = Inf;  active(j) = false;  sz(i) = sz(i) + sz(j);
        ids(i) = n + k;  members{i} = [members{i} members{j}];
    end
    order = members{find(active, 1)};
end

function drawBox(ax, x, v, col, w, alpha, C)
    v = v(~isnan(v));  if isempty(v), return; end
    q1 = qtile(v, 0.25);  q2 = median(v);  q3 = qtile(v, 0.75);  iq = q3 - q1;
    lw = min(v(v >= q1 - 1.5*iq));  hw = max(v(v <= q3 + 1.5*iq));
    plot(ax, [x x], [lw q1], 'Color', C.muted, 'LineWidth', 0.6);
    plot(ax, [x x], [q3 hw], 'Color', C.muted, 'LineWidth', 0.6);
    patch(ax, x + [-w w w -w]/2, [q1 q1 q3 q3], col, 'FaceAlpha', alpha, 'EdgeColor', col);
    if alpha >= 1, mc = [1 1 1]; else, mc = C.ink; end
    plot(ax, x + [-w w]/2, [q2 q2], 'Color', mc, 'LineWidth', 1.2);
end

function drawViolin(ax, x, v, col, halfW)
    g = linspace(min(v), max(v), 200);  f = kde(v, g);  f = f / max(f) * halfW;
    patch(ax, [x + f, fliplr(x - f)], [g, fliplr(g)], col, 'EdgeColor', 'none', 'FaceAlpha', 0.85);
end

function drawDonut(ax, vals, cols, inner)
    vals = vals / sum(vals);  a0 = pi/2;
    for i = 1:numel(vals)
        if vals(i) <= 0, continue; end
        a1 = a0 - vals(i)*2*pi;  t = linspace(a0, a1, max(8, ceil(vals(i)*200)));
        patch(ax, [cos(t) inner*cos(fliplr(t))], [sin(t) inner*sin(fliplr(t))], cols(i,:), ...
              'EdgeColor', 'w', 'LineWidth', 1);
        a0 = a1;
    end
end

function scatterCalls(ax, xv, yv, call, C)
    scatter(ax, xv(call == 0), yv(call == 0), 3, C.ns, 'filled', 'MarkerFaceAlpha', 0.6);
    scatter(ax, xv(call == -1), yv(call == -1), 6, C.down, 'filled');
    scatter(ax, xv(call == 1), yv(call == 1), 6, C.up, 'filled');
end

function classLegend(ax, names, cols, loc)
    h = gobjects(numel(names), 1);
    for i = 1:numel(names), h(i) = patch(ax, NaN, NaN, cols(i,:), 'EdgeColor', 'none'); end
    legend(ax, h, names, 'Location', loc, 'Box', 'off');
end

function lbl = colorLabels(labels, groups, COL)
    lbl = arrayfun(@(i) sprintf('\\color[rgb]{%.3f,%.3f,%.3f}%s', COL(groups(i),:), labels{i}), ...
                   1:numel(labels), 'UniformOutput', false);
end

function padLimits(ax, f)
    xl = xlim(ax);  yl = ylim(ax);
    xlim(ax, xl + [-1 1]*f*diff(xl));  ylim(ax, yl + [-1 1]*f*diff(yl));
end

function fig = newFig(wmm, hmm)
    fig = figure('Units', 'centimeters', 'Position', [1 1 wmm/10 hmm/10], 'Color', 'w', 'Visible', 'off');
    set(fig, 'DefaultAxesFontName', 'Arial', 'DefaultTextFontName', 'Arial', ...
        'DefaultAxesFontSize', 6.5, 'DefaultTextFontSize', 6.5, 'DefaultLegendFontSize', 6.5, ...
        'DefaultAxesLineWidth', 0.6, 'DefaultAxesTickDir', 'out', 'DefaultAxesBox', 'off', ...
        'DefaultAxesTitleFontSizeMultiplier', 1.2, 'DefaultAxesLabelFontSizeMultiplier', 1.08);
end

function finishAxes(ax, C)
    ax.XColor = C.muted;  ax.YColor = C.muted;
    ax.XAxis.TickLabelColor = C.ink2;  ax.YAxis.TickLabelColor = C.ink2;
    ax.XLabel.Color = C.ink;  ax.YLabel.Color = C.ink;  ax.Title.Color = C.ink;
    ax.GridColor = C.grid;  ax.GridAlpha = 1;  ax.Layer = 'top';
end

function hideAxisLines(ax, C, hx, hy)
    if hx, ax.XAxis.Color = 'none';  ax.XAxis.TickLabelColor = C.ink2; end
    if hy, ax.YAxis.Color = 'none';  ax.YAxis.TickLabelColor = C.ink2; end
    ax.XLabel.Color = C.ink;  ax.YLabel.Color = C.ink;
end

function saveFig(fig, OUT, PDF, name)
    set(findall(fig, 'Type', 'Legend'), 'ItemTokenSize', [10 7]);
    exportgraphics(fig, fullfile(OUT, [name '.png']), 'Resolution', 300);
    exportgraphics(fig, PDF, 'Append', true);
    close(fig);
    fprintf('  saved %s.png\n', name);
end

function rgb = hex2rgb(h)
    h = char(h);  rgb = [hex2dec(h(2:3)) hex2dec(h(4:5)) hex2dec(h(6:7))] / 255;
end

function cmap = makeCmap(hexes)
    stops = cell2mat(cellfun(@hex2rgb, hexes, 'UniformOutput', false)');
    cmap = interp1(linspace(0, 1, size(stops,1)), stops, linspace(0, 1, 256));
end
