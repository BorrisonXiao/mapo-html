#!/usr/bin/env python3
"""Build the MAPO GitHub Pages site and attention-normalization analysis."""

from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "data/attention-normalization-summary.json").read_text(encoding="utf-8"))
ASSETS = ROOT / "assets/attention"


def moving_average(values: list[float], width: int = 7) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if len(array) < width:
        return array
    left = width // 2
    padded = np.pad(array, (left, width - left - 1), mode="edge")
    return np.convolve(padded, np.ones(width) / width, mode="valid")


def plot_qwen() -> None:
    qwen = DATA["qwen"]
    x = np.arange(1, qwen["response_tokens"] + 1)
    mean_mass = np.asarray(qwen["mean_mass"])
    max_mass = np.asarray(qwen["max_mass"])
    uniform = np.asarray(qwen["uniform"])
    mean_lift = mean_mass / uniform
    max_lift = max_mass / uniform
    boundary = qwen["caption_boundary_token"]

    plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(2, 1, figsize=(11.4, 7.1), sharex=True)
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(axis="y", color="#d9dedb", linewidth=0.75, alpha=0.8)
        ax.axvline(boundary, color="#8a5a00", linestyle="--", linewidth=1.4)

    axes[0].plot(x, uniform, color="#67726a", linestyle=":", linewidth=2.2, label="Uniform-key baseline A/Nₜ")
    axes[0].plot(x, moving_average(mean_mass), color="#176b4d", linewidth=2.0, label="Mean-head audio mass (7-token mean)")
    axes[0].plot(x, moving_average(max_mass), color="#e67e22", linewidth=1.8, label="Max-head audio mass (7-token mean)")
    axes[0].set_ylabel("Audio attention mass")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title("Absolute mass is far below a uniform-by-key null")
    axes[0].legend(loc="upper right", frameon=True, fontsize=9)

    axes[1].plot(x, moving_average(mean_lift), color="#176b4d", linewidth=2.0, label="Mean-head lift")
    axes[1].plot(x, moving_average(max_lift), color="#e67e22", linewidth=1.8, label="Max-head lift")
    axes[1].axhline(1.0, color="#67726a", linestyle=":", linewidth=2.0, label="Uniform per-key allocation")
    axes[1].set_yscale("log")
    axes[1].set_ylim(0.008, 1.35)
    axes[1].set_ylabel("Length-adjusted lift  mₜ / (A/Nₜ)")
    axes[1].set_xlabel("Generated-token index")
    axes[1].set_title("The post-caption decline remains after length adjustment")
    axes[1].legend(loc="upper right", frameon=True, fontsize=9)
    axes[1].annotate(
        "description → choice evaluation",
        xy=(boundary, 0.82),
        xytext=(boundary + 12, 1.02),
        arrowprops={"arrowstyle": "->", "color": "#8a5a00"},
        color="#8a5a00",
        fontsize=9,
    )
    fig.suptitle("Qwen3-Omni MMAR case: context dilution is too smooth to explain the transition", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(ASSETS / "qwen-length-normalization.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_moss() -> None:
    rows = DATA["moss"]
    labels = [row["benchmark"] for row in rows]
    x = np.arange(len(labels))
    width = 0.32
    raw = [row["raw_mean"] for row in rows]
    normalized = [row["normalized_mean"] for row in rows]
    raw_max = [row["raw_max"] for row in rows]
    normalized_max = [row["normalized_max"] for row in rows]

    plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(axis="y", color="#d9dedb", linewidth=0.75, alpha=0.8)
    bars1 = ax.bar(x - width / 2, raw, width, color="#9ebbd4", label="Raw mean-head ratio")
    bars2 = ax.bar(x + width / 2, normalized, width, color="#55a47e", label="Length-adjusted mean-head ratio")
    ax.scatter(x - width / 2, raw_max, color="#245d91", marker="D", s=42, zorder=4, label="Raw max-head ratio")
    ax.scatter(x + width / 2, normalized_max, color="#176b4d", marker="D", s=42, zorder=4, label="Length-adjusted max-head ratio")
    ax.axhline(1, color="#67726a", linestyle="--", linewidth=1.3)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.42)
    ax.set_ylabel("Median late/early attention ratio")
    ax.set_title("Length adjustment weakens—but does not erase—the apparent decay")
    ax.legend(ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False)
    for bars in (bars1, bars2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.035, f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(ASSETS / "moss-normalization-summary.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def nav(prefix: str, active: str) -> str:
    home_current = ' aria-current="page"' if active == "home" else ""
    attention_current = ' aria-current="page"' if active == "attention" else ""
    return f"""
  <nav class="site-nav" aria-label="Primary navigation">
    <div class="nav-inner">
      <a class="brand" href="{prefix}index.html">MAPO · Audio reasoning</a>
      <div class="nav-links">
        <a href="{prefix}index.html"{home_current}>Home</a>
        <a href="{prefix}attention/"{attention_current}>Attention analyses</a>
      </div>
    </div>
  </nav>"""


MATHJAX = r"""
  <script>
    window.MathJax = {
      tex: { inlineMath: [['\\(', '\\)']], displayMath: [['\\[', '\\]']] },
      svg: { fontCache: 'global' }
    };
  </script>
  <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>"""


def shell(title: str, description: str, css: str, body: str, mathjax: bool = False) -> str:
    scripts = MATHJAX if mathjax else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="{html.escape(description, quote=True)}">
  <title>{html.escape(title)}</title>
  <link rel="stylesheet" href="{css}">{scripts}
</head>
<body>{body}
</body>
</html>
"""


def home_page() -> str:
    body = nav("", "home") + """
  <main>
    <header class="hero">
      <div class="container">
        <p class="eyebrow">MAPO · Multimodal reasoning</p>
        <h1>Mechanistic analyses of audio reasoning</h1>
        <p class="lede">A browsable home for evidence on when audio-language models use—or stop using—the acoustic input during chain-of-thought reasoning.</p>
        <div class="button-row"><a class="button primary" href="attention/">Browse attention analyses</a></div>
      </div>
    </header>
    <section class="section">
      <div class="container">
        <p class="eyebrow">Collections</p>
        <h2>Current research pages</h2>
        <div class="card-grid">
          <a class="card card-link" href="attention/">
            <span class="tag green">Attention analyses</span>
            <h3>Audio use across generated reasoning</h3>
            <p>Token-wise audio attention, entropy changes under audio removal, cumulative evidence timing, and controls for growing causal context.</p>
          </a>
        </div>
      </div>
    </section>
  </main>
  <footer class="site-footer"><div class="container">MAPO · Audio-language model analysis · Updated 5 September 2026</div></footer>"""
    return shell(
        "MAPO · Audio reasoning analyses",
        "MAPO research pages on audio attention, entropy, and modality retention in reasoning models.",
        "assets/site.css",
        body,
    )


def collection_page() -> str:
    body = nav("../", "attention") + """
  <main>
    <header class="hero">
      <div class="container">
        <p class="eyebrow">Collection</p>
        <h1>Attention analyses</h1>
        <p class="lede">Controls and case studies for distinguishing ordinary causal-context dilution from an abrupt loss of acoustic routing during generated reasoning.</p>
      </div>
    </header>
    <section class="section">
      <div class="container">
        <p class="eyebrow">Available analyses</p>
        <h2>Current items</h2>
        <div class="card-grid">
          <a class="card card-link" href="normalizing-audio-attention.html">
            <span class="tag green">Analysis 01</span>
            <h3>Should audio attention be normalized for context length?</h3>
            <p>Tests the dilution objection against one Qwen transition case and 24 held-out MOSS examples from MMAU, MMAR, and MMSU, then specifies a reviewer-ready plotting protocol.</p>
            <p class="item-meta">5 September 2026 · Mechanistic diagnostic</p>
          </a>
        </div>
      </div>
    </section>
  </main>
  <footer class="site-footer"><div class="container">MAPO · Attention analyses</div></footer>"""
    return shell(
        "Attention analyses · MAPO",
        "MAPO attention analyses for audio-language model reasoning.",
        "../assets/site.css",
        body,
    )


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def moss_table_rows() -> str:
    return "\n".join(
        f"""<tr>
          <td><strong>{row['benchmark']}</strong></td>
          <td>{fmt(row['raw_mean'])} / {fmt(row['raw_max'])}</td>
          <td>{fmt(row['normalized_mean'])} / {fmt(row['normalized_max'])}</td>
          <td>{row['raw_decay_cases']}/{row['samples']} → {row['normalized_decay_cases']}/{row['samples']}</td>
          <td>{fmt(row['median_expected_last_first'])}</td>
        </tr>"""
        for row in DATA["moss"]
    )


def first_token_rows() -> str:
    q = DATA["qwen"]
    moss = "\n".join(
        f"""<tr><td>MOSS · {row['benchmark']}</td><td>{row['median_audio_tokens']:g}</td><td>{row['median_prompt_tokens']:g}</td><td>{row['median_uniform_first']:.3f}</td><td>{row['median_observed_first_mean']:.4f}</td><td>{row['median_observed_first_max']:.4f}</td></tr>"""
        for row in DATA["moss"]
    )
    return (
        f"""<tr><td>Qwen · MMAR example</td><td>{q['audio_tokens']}</td><td>{q['response_start']}</td><td>{q['first']['uniform']:.3f}</td><td>{q['first']['mean_mass']:.4f}</td><td>{q['first']['max_mass']:.4f}</td></tr>"""
        + moss
    )


def figure(number: int, src: str, alt: str, caption: str, extra_class: str = "") -> str:
    cls = f"analysis-figure {extra_class}".strip()
    return f"""<figure class="{cls}">
      <a href="{src}" target="_blank" rel="noopener"><img src="{src}" alt="{html.escape(alt, quote=True)}" loading="lazy"></a>
      <figcaption><strong>Figure {number}.</strong> {caption}</figcaption>
    </figure>"""


def report_page() -> str:
    q = DATA["qwen"]
    desc = q["windows"]["description"]
    choice = q["windows"]["choice_evaluation"]
    first_nonaudio_keys = q["response_start"] + 1 - q["audio_tokens"]
    first_audio_per_key = q["first"]["mean_mass"] / q["audio_tokens"]
    first_nonaudio_per_key = (1 - q["first"]["mean_mass"]) / first_nonaudio_keys
    first_nonaudio_advantage = first_nonaudio_per_key / first_audio_per_key
    figures = {
        "qwen_normalized": figure(
            1,
            "../assets/attention/qwen-length-normalization.png",
            "Two-panel Qwen MMAR plot comparing raw audio attention with the uniform-key baseline and length-adjusted lift across generated tokens.",
            "Qwen MMAR sample <code>B2UwFhik…</code>. Top: raw audio mass and the smooth uniform-key null. Bottom: length-adjusted lift on a log scale. The approximate description-to-choice boundary is marked at token 138. Seven-token moving averages are shown for readability; calculations use the original per-token values.",
        ),
        "qwen_original": figure(
            2,
            "../assets/attention/qwen-mmar-original.png",
            "Original MAPO Qwen MMAR plot showing token entropy, entropy change when audio is removed, and mean and maximum audio attention mass.",
            "Original MAPO view for “How to make this song less funny?” The audio-attention spike during the model’s invented acoustic description gives way to a lower regime as it evaluates choices; the prediction is wrong. Entropy difference and attention are aligned by generated-token position but measure different phenomena.",
        ),
        "moss_summary": figure(
            3,
            "../assets/attention/moss-normalization-summary.png",
            "Grouped bar chart of median raw and length-adjusted late-to-early attention ratios for MOSS on MMAU, MMAR, and MMSU.",
            "MOSS-Audio-4B-Thinking, eight random examples per benchmark. Bars show mean-head medians and diamonds show max-head medians. The dashed line at 1 indicates equal late and early attention after the stated transformation. No error bars are shown because this is a small diagnostic, not a benchmark estimate.",
        ),
        "mmar_1": figure(
            4,
            "../assets/attention/moss-mmar-auditory-attention.png",
            "MOSS MMAR sample with entropy, entropy delta, raw audio attention, and cumulative token-mass panels.",
            "MMAR: an incorrect auditory-attention judgment. Audio attention has several mid-reasoning peaks rather than a monotonic decline, and its cumulative curve is close to uniform token mass. This is inconsistent with a universal, mechanically forced decay pattern.",
            "compact-figure",
        ),
        "mmar_2": figure(
            5,
            "../assets/attention/moss-mmar-clarinet.png",
            "MOSS MMAR clarinet performance sample with entropy, entropy delta, raw audio attention, and cumulative token-mass panels.",
            "MMAR: clarinet proficiency. The longer 138-token response provides a useful context-growth stress case; cumulative curves make the timing of audio use visible but do not by themselves correct for the changing key count.",
            "compact-figure",
        ),
        "mmau_1": figure(
            6,
            "../assets/attention/moss-mmau-chord.png",
            "MOSS MMAU chord-identification sample with entropy, entropy delta, raw audio attention, and cumulative token-mass panels.",
            "MMAU: a wrong chord decision. Entropy-difference mass accumulates much earlier than audio-attention mass. The disagreement is a reminder that attention is a routing statistic, whereas removal-based entropy change is closer to a functional intervention.",
            "compact-figure",
        ),
        "mmau_2": figure(
            7,
            "../assets/attention/moss-mmau-pronouns.png",
            "MOSS MMAU speech sample with entropy, entropy delta, raw audio attention, and cumulative token-mass panels.",
            "MMAU: a correct speech-token answer. The example adds a speech-focused task to the music case and shows why benchmark-level aggregation should preserve task diversity.",
            "compact-figure",
        ),
        "mmsu_1": figure(
            8,
            "../assets/attention/moss-mmsu-homophone.png",
            "MOSS MMSU near-homophone sample with entropy, entropy delta, raw audio attention, and cumulative token-mass panels.",
            "MMSU: correct near-homophone perception. Most audio-attention mass arrives in a few localized bursts around acoustic interpretation, not simply at the beginning of generation. A scalar late/early ratio hides this structure.",
            "compact-figure",
        ),
        "mmsu_2": figure(
            9,
            "../assets/attention/moss-mmsu-pun.png",
            "MOSS MMSU pun-interpretation sample with entropy, entropy delta, raw audio attention, and cumulative token-mass panels.",
            "MMSU: correct pun interpretation. Together with Figure 8, this illustrates why the normalized curve, cumulative timing curve, and removal control should be read jointly.",
            "compact-figure",
        ),
    }

    body = nav("../", "attention") + fr"""
  <main>
    <header class="hero report-hero">
      <div class="container">
        <p class="eyebrow">Attention analysis 01 · 5 September 2026</p>
        <h1>Should audio attention be normalized for context length?</h1>
        <p class="lede">Yes—but as a control beside absolute mass, not as a replacement. Length correction explains some smooth decline in MOSS; it cannot explain Qwen’s sharper transition after acoustic description.</p>
        <div class="button-row">
          <a class="button primary" href="#verdict">Read the verdict</a>
          <a class="button" href="./">← All attention analyses</a>
          <a class="button" href="#examples">Jump to examples</a>
        </div>
      </div>
    </header>

    <section class="section">
      <div class="container split">
        <aside class="toc" aria-label="Table of contents">
          <strong>On this page</strong>
          <a href="#verdict">Verdict</a>
          <a href="#null">Length-control null</a>
          <a href="#qwen">Qwen transition case</a>
          <a href="#moss">MOSS diagnostic</a>
          <a href="#examples">Held-out examples</a>
          <a href="#protocol">Plotting protocol</a>
          <a href="#limits">Limits and status</a>
        </aside>

        <article class="prose">
          <section id="verdict">
            <p class="eyebrow">Bottom line</p>
            <h2>Plot both the amount and the preference</h2>
            <div class="callout">
              <strong>Recommendation</strong>
              <p>Keep raw audio-attention mass as the primary routing quantity. Add a uniform-key baseline and a length-adjusted lift curve. Mark reasoning-phase boundaries, and decompose the competing context into audio, static prompt text, and generated history.</p>
            </div>
            <p>The reviewer’s denominator argument is real: as a causal response grows, the fixed audio region competes with more generated keys. That mechanism predicts a smooth decline proportional to the audio token share. It does <em>not</em> predict a discrete drop synchronized with the model moving from acoustic description into answer selection.</p>
            <div class="stat-grid">
              <div class="stat"><strong>9/24 → 6/24</strong><span>MOSS strong-decay cases before → after diagnostic length adjustment</span></div>
              <div class="stat"><strong>0.87</strong><span>Qwen first-token uniform audio share in the focal example</span></div>
              <div class="stat"><strong>0.052</strong><span>Observed first-token mean-head audio mass</span></div>
              <div class="stat"><strong>0.515</strong><span>Uniform audio share at the final Qwen response token</span></div>
            </div>
          </section>

          <section id="null" class="section">
            <p class="eyebrow">Reviewer concern</p>
            <h2>A useful null, not a neural expectation</h2>
            <p>For generated position \(t\), let \(m_t\) be the total attention assigned to audio keys, \(A\) the number of valid audio keys, and \(N_t\) the total number of causally available non-padding keys. Uniform attention over available keys would assign:</p>
            <div class="equation">
              \[
              m_t=\sum_{{j\in\mathrm{{audio}}}} a_{{t,j}},
              \qquad
              u_t=\frac{{A}}{{N_t}},
              \qquad
              L_t=\frac{{m_t}}{{u_t}}=\frac{{m_tN_t}}{{A}}.
              \]
            </div>
            <p>The lift \(L_t=1\) denotes uniform allocation per available key; \(L_t&lt;1\) means that audio is underweighted relative to its token count. Uniform attention is only a descriptive null: real heads have content, position, sink, and specialization biases.</p>
            <div class="table-wrap">
              <table>
                <caption><strong>Table 1.</strong> First generated-position audio allocation. Prefix keys count the full processed prompt; the uniform denominator also contains the current response-position key in this teacher-forced trace. MOSS rows report medians over eight examples. The uniform share is a key-count null, not a claim about expected trained-model behavior.</caption>
                <thead><tr><th>System · benchmark</th><th>Audio keys</th><th>Prefix keys</th><th>Uniform share</th><th>Observed mean</th><th>Observed max</th></tr></thead>
                <tbody>{first_token_rows()}</tbody>
              </table>
            </div>
            <p>The first generated position therefore does not strongly attend audio merely because audio occupies much of the history. In every row of Table 1, observed mean-head audio mass is orders of magnitude below the key-count share. In the Qwen case, audio receives {q['first']['mean_mass']:.3f} mass, leaving {1-q['first']['mean_mass']:.3f} for {first_nonaudio_keys} non-audio keys—the static prompt plus the current teacher-forced query position. The average non-audio key receives approximately {first_nonaudio_advantage:.0f}× as much attention as the average audio-region key. This verifies strong non-audio focus, although the present collector cannot yet separate static text-prompt keys from that query position.</p>
          </section>

          <section id="qwen" class="section">
            <p class="eyebrow">Qwen case study</p>
            <h2>The denominator changes gradually; the routing regime changes faster</h2>
            <p>The focal MMAR prompt contains {q['audio_tokens']} detected audio-region keys in a {q['response_start']}-token prefix. Its uniform baseline falls from {q['first']['uniform']:.3f} at the first response position to {q['last_uniform']:.3f} at the last—a gradual {100 * (1 - q['last_uniform'] / q['first']['uniform']):.1f}% reduction across all {q['response_tokens']} generated tokens.</p>
            {figures['qwen_normalized']}
            <p>During the manually aligned description window, tokens 80–124, median mean-head raw mass is {desc['mean_raw_median']:.3f} and normalized lift is {desc['mean_lift_median']:.3f}. During choice evaluation, tokens 145–219, they fall to {choice['mean_raw_median']:.3f} and {choice['mean_lift_median']:.3f}. The normalized measure therefore preserves the qualitative change.</p>
            {figures['qwen_original']}
            <div class="callout amber">
              <strong>Interpretation, not yet a causal verdict</strong>
              <p>The synchronized drop supports a routing-shift hypothesis, but attention alone is not causal attribution. The entropy change under audio removal and answer changes under silence are complementary evidence, not interchangeable measurements.</p>
            </div>
          </section>

          <section id="moss" class="section">
            <p class="eyebrow">MOSS-Audio-4B-Thinking</p>
            <h2>Normalization matters, especially outside MMAR</h2>
            <p>We applied the simple \(m_t/(A/N_t)\) adjustment diagnostically to eight randomly selected examples each from MMAU, MMAR, and MMSU. Early and late values are medians over the first and last quarters of the thinking tokens; each table cell reports mean-head / max-head aggregation.</p>
            {figures['moss_summary']}
            <div class="table-wrap">
              <table>
                <caption><strong>Table 2.</strong> Median late/early audio-attention ratios for MOSS-Audio-4B-Thinking. “Strong decay” requires both mean- and max-head ratios below 0.5. The expected last/first column is the decline produced by key-count dilution alone.</caption>
                <thead><tr><th>Benchmark</th><th>Raw mean / max</th><th>Adjusted mean / max</th><th>Strong-decay cases</th><th>Null last / first</th></tr></thead>
                <tbody>{moss_table_rows()}</tbody>
              </table>
            </div>
            <p>The control changes the story. MMAU’s median adjusted ratio rises above 1, and MMSU moves much closer to parity. MMAR still shows a median adjusted decline and three strong-decay cases out of eight. Length dilution is therefore a partial explanation, not a complete one.</p>
          </section>

          <section id="examples" class="section">
            <p class="eyebrow">Held-out test benchmarks</p>
            <h2>The token-wise shape is not universally decreasing</h2>
            <p>Figures 4–9 sample MMAU, MMAR, and MMSU rather than AVQA. Each source plot contains raw token-wise entropy, the entropy difference between text-only and audio-conditioned scoring, mean/max audio-attention mass, and a CDF-like cumulative row. The cumulative row answers <em>when</em> mass accumulates; it is not a correction for context length.</p>
            {figures['mmar_1']}
            {figures['mmar_2']}
            {figures['mmau_1']}
            {figures['mmau_2']}
            {figures['mmsu_1']}
            {figures['mmsu_2']}
          </section>

          <section id="protocol" class="section">
            <p class="eyebrow">Recommended plotting standard</p>
            <h2>One figure should answer four different questions</h2>
            <ol class="steps">
              <li><strong>Absolute routing</strong><p>Keep raw audio mass \(m_t\). It retains a direct probability-mass interpretation and shows how much attention reaches the audio region.</p></li>
              <li><strong>Length control</strong><p>Overlay \(A/N_t\), then add lift \(L_t=m_t/(A/N_t)\) with a reference line at 1. Count valid causal keys from the actual attention mask.</p></li>
              <li><strong>Competition destination</strong><p>Report audio, static prompt text, and generated-prefix mass separately, together with their per-key densities. This reveals where attention goes after an acoustic-description phase ends.</p></li>
              <li><strong>Functional control</strong><p>Keep the real-audio versus silence entropy/log-probability difference and final-answer change. Attention is routing evidence; intervention measures whether the audio changes computation.</p></li>
            </ol>
            <p>A symmetric per-key comparison is useful as a secondary statistic:</p>
            <div class="equation">
              \[
              R_t=
              \frac{{m_t/A}}{{(1-m_t)/(N_t-A)}}.
              \]
            </div>
            <p>Plot \(\log R_t\) with a zero reference line. Positive values favor audio per key; negative values favor the non-audio context. This is more interpretable than dividing audio mass by the number of audio tokens alone.</p>
            <div class="table-wrap">
              <table>
                <caption><strong>Table 3.</strong> Proposed measurements and the distinct question answered by each. No single row is a substitute for the others.</caption>
                <thead><tr><th>Quantity</th><th>Question answered</th><th>Reference</th><th>Primary caveat</th></tr></thead>
                <tbody>
                  <tr><td>Raw audio mass \(m_t\)</td><td>How much routing reaches audio?</td><td>0 to 1</td><td>Changes as the competing key set grows</td></tr>
                  <tr><td>Lift \(L_t\)</td><td>Is audio favored relative to its key share?</td><td>1 = uniform per key</td><td>Uniform is a null, not a trained-model expectation</td></tr>
                  <tr><td>Log odds lift \(\log R_t\)</td><td>Audio versus non-audio per-key preference?</td><td>0 = equal per-key density</td><td>Needs clipping near mass 0 or 1</td></tr>
                  <tr><td>Cumulative token mass</td><td>When does the sequence spend its total mass?</td><td>Diagonal = uniform over response positions</td><td>Not a context-length correction</td></tr>
                  <tr><td>Audio-removal entropy delta</td><td>Does audio alter token prediction?</td><td>0 = no entropy change</td><td>Silence is an intervention with its own distribution shift</td></tr>
                </tbody>
              </table>
            </div>
          </section>

          <section id="limits" class="section">
            <p class="eyebrow">Status and limitations</p>
            <h2>What is measured, and what remains to implement</h2>
            <ul>
              <li>The MOSS study contains 24 examples—eight per test benchmark, selected randomly with seed 17—and should not be read as a benchmark confidence interval.</li>
              <li>The MOSS normalization in Table 2 was applied to already aggregated mean/max curves as a diagnostic. The production implementation should normalize every layer/head first, then reduce. This matters most for max-head analysis.</li>
              <li>Qwen’s audio-region span follows the existing token-ID detector and includes the model’s audio wrapper region. The primary production mask should count actual audio-feature tokens, with wrappers and time markers reported as a sensitivity analysis.</li>
              <li>The Qwen caption boundary at token {q['caption_boundary_token']} is manually aligned to “Next, evaluating the choices.” Automated phase annotation or a registered change-point test should replace it for aggregate claims.</li>
              <li>MOSS values use decoder layers 28–35. Qwen’s focal curve averages all 48 decoder layers. Layer-resolved reporting should accompany any cross-model claim.</li>
            </ul>
            <details>
              <summary>Reproducibility snapshot</summary>
              <p>Source Qwen run: <code>{html.escape(q['source_run'])}</code></p>
              <p>Source MOSS runs: <code>exp/analysis/{{mmau,mmar,mmsu}}/MOSS-Audio-4B-Thinking/v1-20260905-entropy-cdf</code></p>
              <p>The site repository contains the compact numeric snapshot, selected published figures, and scripts that rebuild this page and the two summary plots.</p>
            </details>
          </section>
        </article>
      </div>
    </section>
  </main>
  <footer class="site-footer"><div class="container">MAPO · Small mechanistic diagnostic using Qwen3-Omni and MOSS-Audio-4B-Thinking artifacts.</div></footer>"""
    return shell(
        "Normalizing audio attention · MAPO",
        "Analysis of context-length normalization for audio attention in Qwen3-Omni and MOSS-Audio-4B-Thinking across MMAU, MMAR, and MMSU.",
        "../assets/site.css",
        body,
        mathjax=True,
    )


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    plot_qwen()
    plot_moss()
    (ROOT / "index.html").write_text(home_page(), encoding="utf-8")
    (ROOT / "attention/index.html").write_text(collection_page(), encoding="utf-8")
    (ROOT / "attention/normalizing-audio-attention.html").write_text(report_page(), encoding="utf-8")


if __name__ == "__main__":
    main()
