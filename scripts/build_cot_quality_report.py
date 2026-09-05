#!/usr/bin/env python3
"""Build the MOSS/AF-Next chain-of-thought quality comparison page."""

from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from build_attention_analysis import nav, shell


REPORT_ROOT = Path(__file__).resolve().parents[1]
SITE_ROOT = REPORT_ROOT
DATA = json.loads(
    (REPORT_ROOT / "data/cot-quality-summary.json").read_text(encoding="utf-8")
)
ASSETS = SITE_ROOT / "assets/cot-quality"
COLORS = {"MOSS-4B": "#7599b8", "MOSS-8B": "#245d91", "AF-Next": "#d77a2b"}


def plot_quality_summary() -> None:
    models = DATA["models"]
    labels = [model["short"] for model in models]
    x = np.arange(len(models))
    dimensions = ["grounding", "logic", "correctness", "consistency", "economy"]
    dimension_labels = ["Grounding", "Logic", "Correctness", "Consistency", "Economy"]
    dimension_colors = ["#4c78a8", "#72a0c1", "#55a47e", "#f2b35d", "#d77a2b"]

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.4))
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(axis="y", color="#d9dedb", linewidth=0.7, alpha=0.8)

    bottom = np.zeros(len(models))
    for key, label, color in zip(dimensions, dimension_labels, dimension_colors):
        values = np.array([model["dimensions"][key] for model in models])
        axes[0].bar(x, values, bottom=bottom, color=color, width=0.62, label=label)
        bottom += values
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 8.4)
    axes[0].set_ylabel("Mean audit score (0–8)")
    axes[0].set_title("Trace utility")
    for index, value in enumerate(bottom):
        axes[0].text(index, value + 0.13, f"{value:.2f}", ha="center", fontsize=9)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncols=2, frameon=False, fontsize=8)

    correct = [model["correct"] for model in models]
    axes[1].bar(x, correct, color=[COLORS[label] for label in labels], width=0.62)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0, 24)
    axes[1].set_ylabel("Reference-consistent answers / 24")
    axes[1].set_title("Small-sample correctness")
    for index, value in enumerate(correct):
        axes[1].text(index, value + 0.45, f"{value}/24", ha="center", fontsize=9)

    lengths = [model["median_completion_tokens"] for model in models]
    axes[2].bar(x, lengths, color=[COLORS[label] for label in labels], width=0.62)
    axes[2].set_xticks(x, labels)
    axes[2].set_ylim(0, max(lengths) * 1.2)
    axes[2].set_ylabel("Median generated tokens")
    axes[2].set_title("Reasoning cost")
    for index, value in enumerate(lengths):
        axes[2].text(index, value + 4, f"{value:g}", ha="center", fontsize=9)

    fig.suptitle("Longer traces did not become more useful on the 24-example audit", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(ASSETS / "cot-quality-summary.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_attention_summary() -> None:
    rows = DATA["attention"]
    labels = [f"{row['model_short']}\n{row['benchmark']}" for row in rows]
    x = np.arange(len(rows))
    width = 0.34
    raw = [row["raw_mean_median_ratio"] for row in rows]
    adjusted = [row["adjusted_mean_median_ratio"] for row in rows]
    raw_cases = [row["raw_strong_decay_cases"] for row in rows]
    adjusted_cases = [row["adjusted_strong_decay_cases"] for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(12.0, 7.1), sharex=True)
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(axis="y", color="#d9dedb", linewidth=0.7, alpha=0.8)

    axes[0].bar(x - width / 2, raw, width, color="#9ebbd4", label="Raw mean-head ratio")
    axes[0].bar(x + width / 2, adjusted, width, color="#55a47e", label="Length-adjusted mean-head ratio")
    axes[0].axhline(1, color="#67726a", linestyle="--", linewidth=1.2)
    axes[0].axhline(0.5, color="#8a5a00", linestyle=":", linewidth=1.1)
    axes[0].set_ylabel("Median late / early ratio")
    axes[0].set_title("Length correction helps, but AF-Next remains below 0.5 on every benchmark")
    axes[0].legend(frameon=False, ncols=2, loc="upper right")

    axes[1].bar(x - width / 2, raw_cases, width, color="#9ebbd4", label="Raw")
    axes[1].bar(x + width / 2, adjusted_cases, width, color="#55a47e", label="Length-adjusted")
    axes[1].set_ylim(0, 8.5)
    axes[1].set_ylabel("Strong-decay cases / 8")
    axes[1].set_xticks(x, labels)
    axes[1].set_title("Per-example criterion: both mean- and max-head late/early ratios < 0.5")
    axes[1].legend(frameon=False, ncols=2, loc="upper right")
    for index, (a, b) in enumerate(zip(raw_cases, adjusted_cases)):
        axes[1].text(index - width / 2, a + 0.15, str(a), ha="center", fontsize=8)
        axes[1].text(index + width / 2, b + 0.15, str(b), ha="center", fontsize=8)

    fig.suptitle("The larger thinking checkpoints show stronger late-stage audio-attention decay", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(ASSETS / "attention-decay-comparison.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig(number: int, src: str, alt: str, caption: str) -> str:
    return f"""<figure class="analysis-figure compact-figure">
      <a href="{src}" target="_blank" rel="noopener"><img src="{src}" alt="{html.escape(alt, quote=True)}" loading="lazy"></a>
      <figcaption><strong>Figure {number}.</strong> {caption}</figcaption>
    </figure>"""


def model_rows() -> str:
    return "\n".join(
        f"""<tr>
          <td><strong>{model['short']}</strong></td>
          <td>{model['mean_total_score']:.2f} / 8</td>
          <td>{model['correct']}/24</td>
          <td>{model['median_completion_tokens']:g}</td>
          <td>{model['dimensions']['grounding']:.2f} / 2</td>
          <td>{model['dimensions']['logic']:.2f} / 2</td>
          <td>{model['dimensions']['economy']:.2f} / 1</td>
        </tr>"""
        for model in DATA["models"]
    )


def benchmark_rows() -> str:
    output = []
    for model in DATA["models"]:
        for row in model["benchmark_summary"]:
            output.append(
                f"<tr><td><strong>{model['short']}</strong></td><td>{row['benchmark']}</td>"
                f"<td>{row['mean_total_score']:.2f} / 8</td><td>{row['correct']}/{row['samples']}</td>"
                f"<td>{row['median_completion_tokens']:g}</td></tr>"
            )
    return "\n".join(output)


def attention_rows() -> str:
    output = []
    for model in DATA["models"]:
        model_attention = [row for row in DATA["attention"] if row["model"] == model["name"]]
        for row in model_attention:
            output.append(
                f"<tr><td><strong>{model['short']}</strong></td><td>{row['benchmark']}</td>"
                f"<td>{row['raw_mean_median_ratio']:.2f} / {row['raw_max_median_ratio']:.2f}</td>"
                f"<td>{row['adjusted_mean_median_ratio']:.2f} / {row['adjusted_max_median_ratio']:.2f}</td>"
                f"<td>{row['raw_strong_decay_cases']}/{row['samples']} → {row['adjusted_strong_decay_cases']}/{row['samples']}</td></tr>"
            )
    return "\n".join(output)


def trace_text(response: str) -> str:
    return html.escape(response.strip()).replace("\n", "<br>")


def examples() -> str:
    blocks = []
    for index, row in enumerate(DATA["selected_examples"], start=3):
        score = row["total_score"]
        tags = " · ".join(tag.replace("_", " ") for tag in row["tags"])
        blocks.append(f"""
        <section class="trace-card">
          <div class="trace-heading">
            <div><span class="tag green">{row['model_short']} · {row['benchmark']}</span><h3>{html.escape(row['id'])}</h3></div>
            <strong class="score-pill">{score}/8</strong>
          </div>
          <p><strong>Question.</strong> {html.escape(' '.join(row['prompt'].split()))}</p>
          <p><strong>Reference.</strong> {html.escape(str(row['gold']))}<br><strong>Model answer.</strong> {html.escape(str(row['final_answer']))}</p>
          <div class="callout amber"><strong>Audit</strong><p>{html.escape(row['review_note'])}</p><p class="item-meta">Tags: {html.escape(tags)}</p></div>
          <details><summary>Read the full reasoning trace</summary><div class="trace-text">{trace_text(row['response'])}</div></details>
          {fig(index, row['figure'], f"Token-wise entropy and audio-attention plot for {row['model_short']} {row['benchmark']} sample {row['id']}.", f"{row['model_short']} · {row['benchmark']}. The source plot pairs token entropy and text-minus-audio entropy change with raw mean/max audio-attention mass and a CDF-like cumulative-mass row. Source: <code>{html.escape(row['source_plot'])}</code>.")}
        </section>""")
    return "\n".join(blocks)


def report_page() -> str:
    models = {model["short"]: model for model in DATA["models"]}
    prompt_rows = "\n".join(
        f"""<tr><td><strong>{model['short']}</strong><br><span class="item-meta">{model['parameters']}</span></td>
        <td>{html.escape(model['prompt'])}</td><td>{html.escape(model['generation'])}</td>
        <td><a href="{model['source']}">official model card</a></td></tr>"""
        for model in DATA["models"]
    )
    rubric_rows = "\n".join(
        f"<tr><td><strong>{item['key'].title()}</strong></td><td>0–{item['max']}</td><td>{html.escape(item['definition'])}</td></tr>"
        for item in DATA["rubric"]
    )
    body = nav("../", "attention") + f"""
  <main>
    <header class="hero report-hero">
      <div class="container">
        <p class="eyebrow">Attention analysis 02 · 5 September 2026</p>
        <h1>Are MOSS-8B and AF-Next’s reasoning traces usable?</h1>
        <p class="lede">Not yet, on this diagnostic. MOSS-8B is longer but less useful than the 4B baseline; AF-Next is marginally better than MOSS-8B, yet its timestamped chains often invent evidence. Both larger checkpoints show substantial late-stage audio-attention decay.</p>
        <div class="button-row"><a class="button primary" href="#verdict">Read the verdict</a><a class="button" href="./">← All attention analyses</a><a class="button" href="#examples">Inspect examples</a></div>
      </div>
    </header>
    <section class="section">
      <div class="container split">
        <aside class="toc" aria-label="Table of contents">
          <strong>On this page</strong>
          <a href="#verdict">Verdict</a><a href="#protocol">Prompts & rubric</a><a href="#quality">CoT quality</a><a href="#retention">Modality retention</a><a href="#failures">Failure modes</a><a href="#examples">Nine examples</a><a href="#limits">Limits</a>
        </aside>
        <article class="prose">
          <section id="verdict">
            <p class="eyebrow">Bottom line</p><h2>Neither larger checkpoint fixes the CoT problem</h2>
            <div class="callout"><strong>Recommendation</strong><p>Do not select MOSS-8B or AF-Next for MAPO solely because they emit longer thinking traces. If a small-backbone port proceeds, AF-Next is the more interesting stress case for modality retention, but its visible CoT should be treated as an unreliable explanation rather than faithful evidence.</p></div>
            <div class="stat-grid">
              <div class="stat"><strong>{models['MOSS-4B']['mean_total_score']:.2f}/8</strong><span>MOSS-4B trace utility</span></div>
              <div class="stat"><strong>{models['MOSS-8B']['mean_total_score']:.2f}/8</strong><span>MOSS-8B trace utility</span></div>
              <div class="stat"><strong>{models['AF-Next']['mean_total_score']:.2f}/8</strong><span>AF-Next trace utility</span></div>
              <div class="stat"><strong>{models['AF-Next']['out_of_range_timestamp_traces']}/24</strong><span>AF traces with impossible timestamps</span></div>
            </div>
            <p>The audit covers the same 24 examples per model: eight randomly selected with seed 17 from each of MMAU, MMAR, and MMSU. These are qualitative diagnostics, not published benchmark estimates.</p>
          </section>

          <section id="protocol" class="section">
            <p class="eyebrow">Author-intended use</p><h2>Prompting follows each checkpoint’s documented regime</h2>
            <p>MOSS’s authors pass a task prompt directly to the audio processor when using the Thinking checkpoint; their audio path does not require a separate thinking switch. AF-Next’s authors explicitly recommend a step-by-step, timestamp-grounded instruction and show repetition penalty 1.2. We use their exact short cue below. This asymmetry is intentional: it tests each model as documented, not under an invented common prompt.</p>
            <div class="table-wrap"><table><caption><strong>Table 1.</strong> Prompt and decoding protocol. All new MOSS-8B and AF-Next runs are greedy and permit 512 new tokens; MOSS-4B is the previously generated 256-token reference.</caption><thead><tr><th>Model</th><th>Prompt regime</th><th>Decoding</th><th>Source</th></tr></thead><tbody>{prompt_rows}</tbody></table></div>
            <p>The single-reviewer audit uses five dimensions. Correctness is deliberately separated from grounding and logic: a trace can land on the right option for the wrong reason.</p>
            <div class="table-wrap"><table><caption><strong>Table 2.</strong> Manual CoT quality rubric, totaling eight points. Every one of the 72 traces has a saved rating and note in the aggregate analysis directory.</caption><thead><tr><th>Dimension</th><th>Range</th><th>Operational definition</th></tr></thead><tbody>{rubric_rows}</tbody></table></div>
          </section>

          <section id="quality" class="section">
            <p class="eyebrow">Trace audit</p><h2>More tokens mostly buy more confident prose</h2>
            {fig(1, '../assets/cot-quality/cot-quality-summary.png', 'Three-panel summary of mean chain-of-thought audit scores, reference-consistent answers, and median generated token counts for MOSS-4B, MOSS-8B, and AF-Next.', 'The 24-example audit. Stacked bars preserve the five rubric components; the middle panel reports reference-consistent final choices, not benchmark accuracy; the right panel shows completion length. MOSS-8B and AF-Next both have a 208-token median, versus 93.5 for MOSS-4B.')}
            <div class="table-wrap"><table><caption><strong>Table 3.</strong> Aggregate trace-quality audit. Grounding and logic are scored out of two; economy is scored out of one. The sample is paired across models.</caption><thead><tr><th>Model</th><th>Mean score</th><th>Correct</th><th>Median tokens</th><th>Grounding</th><th>Logic</th><th>Economy</th></tr></thead><tbody>{model_rows()}</tbody></table></div>
            <div class="table-wrap"><table><caption><strong>Table 4.</strong> Breakdown by held-out benchmark. Each row contains eight paired examples.</caption><thead><tr><th>Model</th><th>Benchmark</th><th>Mean score</th><th>Correct</th><th>Median tokens</th></tr></thead><tbody>{benchmark_rows()}</tbody></table></div>
            <p>MOSS-8B is not a quality upgrade in this slice: it answers 15/24 correctly versus 17/24 for MOSS-4B, and its economy score is zero on every trace. AF-Next reaches 16/24, but its modest correctness advantage over MOSS-8B does not translate into trustworthy explanations.</p>
          </section>

          <section id="retention" class="section">
            <p class="eyebrow">Audio routing</p><h2>The decay survives the context-length objection</h2>
            <p>For each thinking span, raw late/early audio-attention ratios are supplemented by the length-adjusted lift <code>audio_mass / (audio_tokens / available_keys)</code>. A strong-decay case requires both the mean-head and max-head ratios to remain below 0.5.</p>
            {fig(2, '../assets/cot-quality/attention-decay-comparison.png', 'Two-panel comparison of raw and length-adjusted late-to-early audio-attention ratios and strong-decay counts across three models and three benchmarks.', 'Length adjustment raises most late/early ratios, as expected. It reduces strong-decay counts from 9 to 6 for MOSS-4B, 14 to 8 for MOSS-8B, and only 17 to 15 for AF-Next. Thus ordinary key-count dilution is not sufficient to explain AF-Next’s pattern.')}
            <div class="table-wrap"><table><caption><strong>Table 5.</strong> Median late/early attention ratios (mean head / max head), followed by strong-decay cases before → after length adjustment. Each benchmark has eight examples.</caption><thead><tr><th>Model</th><th>Benchmark</th><th>Raw ratio</th><th>Adjusted ratio</th><th>Strong decay</th></tr></thead><tbody>{attention_rows()}</tbody></table></div>
            <p>The intervention controls prevent an overly strong conclusion. Silent audio changes the final response in {models['MOSS-8B']['silent_answer_changes']}/24 MOSS-8B cases and {models['AF-Next']['silent_answer_changes']}/24 AF-Next cases. AF-Next is therefore not simply ignoring audio from the start; it often uses audio functionally while still reallocating attention sharply over its generated trace.</p>
          </section>

          <section id="failures" class="section">
            <p class="eyebrow">Failure taxonomy</p><h2>The two larger models fail differently</h2>
            <ul>
              <li><strong>MOSS-8B: procedural boilerplate.</strong> Twenty-one of 24 traces are explicitly tagged for repeated setup language, and the remaining long traces are still uneconomical. It frequently says it will “analyze” or “pinpoint” the answer, then substitutes an unsupported label.</li>
              <li><strong>MOSS-8B: modality leakage.</strong> Three MMAR traces appeal to what “the video shows” even though this run supplies audio only. That is especially damaging for an explanation intended to demonstrate acoustic grounding.</li>
              <li><strong>AF-Next: timestamp-shaped hallucination.</strong> It emits timestamp tags in {models['AF-Next']['timestamp_traces']}/24 traces, but {models['AF-Next']['out_of_range_timestamp_traces']}/24 extend beyond the known clip duration. Other traces invent numbered “segments” or detailed event sequences not warranted by the question.</li>
              <li><strong>Both: right answer, invalid chain.</strong> AF-Next counts nine chops, derives ten pieces, then answers nine. MOSS-8B explicitly calls the gold fruit proposition incorrect, then selects it because the other options are unrelated.</li>
            </ul>
          </section>

          <section id="examples" class="section">
            <p class="eyebrow">Paired qualitative evidence</p><h2>Nine representative traces and token-wise plots</h2>
            <p>The following cases include one failure and one stronger control per family across the three test suites. The full audit covers all 72 traces, while every new model directory contains eight plots per benchmark.</p>
{examples()}
          </section>

          <section id="limits" class="section">
            <p class="eyebrow">Status and limits</p><h2>What this does—and does not—establish</h2>
            <ul>
              <li>This is a deterministic 24-example slice, not a confidence interval or substitute for the authors’ full benchmark evaluations.</li>
              <li>Grounding scores inspect the trace against the prompt, reference, answer, and clip duration. They do not claim an independent human transcription of every waveform.</li>
              <li>AF-Next was given the exact recommended timestamp prompt; impossible timestamps are therefore an observed failure under intended prompting, but a non-timestamp prompt may change its style.</li>
              <li>Attention uses eager decoder attentions from the last eight layers. The mean/max audio mass excludes wrapper tokens, and normalization is applied token-wise before early/late reduction.</li>
              <li>The current attention arrays align a generated token with the decoder row at that token position; causal attribution still requires intervention measures such as the retained real-versus-silence controls.</li>
            </ul>
            <details><summary>Reproducibility snapshot</summary><p>Aggregate audit: <code>exp/analysis/audio-thinking-backbones/v1-20260905-cot-quality</code></p><p>Full runs: <code>exp/analysis/{{mmau,mmar,mmsu}}/{{MOSS-Audio-4B-Thinking,MOSS-Audio-8B-Thinking,Audio-Flamingo-Next-Think}}/v1-20260905-entropy-cdf</code></p><p>Selection: random, seed 17, eight examples per benchmark. New runs: one H100 each, bfloat16 weights, eager attention, last eight decoder layers.</p></details>
          </section>
        </article>
      </div>
    </section>
  </main>
  <footer class="site-footer"><div class="container">MAPO · Paired small-sample audit of MOSS-Audio and Audio Flamingo Next thinking checkpoints.</div></footer>"""
    return shell(
        "CoT quality: MOSS-8B vs AF-Next · MAPO",
        "Paired chain-of-thought quality and modality-retention audit of MOSS-Audio-4B/8B-Thinking and Audio Flamingo Next Think on MMAU, MMAR, and MMSU.",
        "../assets/site.css",
        body,
    )


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    plot_quality_summary()
    plot_attention_summary()
    (SITE_ROOT / "attention").mkdir(parents=True, exist_ok=True)
    (SITE_ROOT / "attention/cot-quality-comparison.html").write_text(
        report_page(), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
