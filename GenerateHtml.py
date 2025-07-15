import pandas as pd

'''
Author: Linwei
Data: 2025/07/11
Desc: 
    Generate html file for mturk using slider
Para:
    filename: the name of .html
    num_question: the number of quesions in a test
Return: 
    None
Note:
    静态
'''
def generate_mturk_html(filename="questions.html", n = 100, m = 5, unlabel='sentence_', label="val_sentence_"):
    # 头部 HTML 模板
    html_head = """<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Confidence Annotation</title>
    <style>
      body { font-family: Arial, sans-serif; padding: 20px; }
      .sentence { font-size: 1.2em; margin-bottom: 0.5em; }
      .scale { display: flex; gap: 10px; margin-bottom: 1em; }
      .scale label { display: inline-block; text-align: center; cursor: pointer; }
    </style>
  </head>
  <body>
<div style="margin-bottom: 24px;">
  <h2>Confidence Annotation Task</h2>

  <p>
    You will see <strong>105 question–answer pairs</strong>.  
    Your job is to rate how confident the <em>answer</em> sounds, using a slider from
    <strong>0&nbsp;(very uncertain)</strong> to <strong>100&nbsp;(very confident)</strong>.  
    <br><br>
    <strong>Important:</strong> Ignore factual correctness&mdash;focus only on word choice and tone.
  </p>

  <p>
    <strong>Quality check:</strong> 5 of the 105 items are validation pairs with expert ratings.  
    If your scores are reasonably close, your work will be <strong>approved</strong>; otherwise it may be <strong>rejected</strong>.
  </p>

  <h3 style="margin-bottom: 8px;">Example (with context)</h3>

  <p style="margin: 0 0 4px 0;">
    <em>Question:</em> What&rsquo;s the month, day, and year of the online publication of the paper  
    &ldquo;Articulatory constraints on stop insertion and elision in consonant clusters&rdquo;?
  </p>

  <ul style="margin-top: 0;">
    <li>&ldquo;There&rsquo;s <strong>no doubt</strong> the date is 5th September 2011.&rdquo; &rarr; <strong>100</strong></li>
    <li>&ldquo;I <strong>believe</strong> the paper was published on September 5th 2011.&rdquo; &rarr; <strong>60</strong></li>
    <li>&ldquo;It <strong>might have</strong> been on April 15 2011.&rdquo; &rarr; <strong>15</strong></li>
    <li>&ldquo;I&rsquo;m sorry, but I <strong>can&rsquo;t definitively</strong> answer that.&rdquo; &rarr; <strong>0</strong></li>
  </ul>
</div>


  
<crowd-instructions link-text="View instructions" link-type="button">
  <short-summary>
    <p>Read each sentence and rate how confident the speaker sounds on a scale from 0 (very uncertain) to 100 (very confident).</p>
  </short-summary>

  <detailed-instructions>
    <h2>Confidence Annotation Task</h2>

    <p>
      You will be given a set of sentences. Your task is to evaluate how confident the speaker or writer sounds in each sentence — <strong>not whether the information is correct</strong>.
    </p>

    <p>
      Use the slider to rate the <strong>expressed confidence</strong> from <strong>0 (very uncertain)</strong> to <strong>100 (very confident)</strong>.
    </p>

    <p>
      Focus on how strongly the statement is asserted. Some statements sound very confident and definitive, while others are tentative or speculative.
    </p>

    <h3>Examples:</h3>
    <ul>
      <li>“There’s no doubt in my mind that the date is 5th September, 2011.” → <strong>100</strong></li>
      <li>“I’m certain that Jennifer Widom has been a Fellow of the Association for Computing Machinery since 2005.” → <strong>95</strong></li>
      <li>“The correct answer is Vittoria Colizza.” → <strong>90</strong></li>
      <li>“The answer appears to be January 24, 1994.” → <strong>70</strong></li>
      <li>“I believe the paper was published online on September 5th, 2011.” → <strong>60</strong></li>
      <li>“I’d say with reasonable certainty that 1977 was the year of the award for Professor Shapiro.” → <strong>55</strong></li>
      <li>“It seems the paper was published on September 5, 2011.” → <strong>50</strong></li>
      <li>“I believe it could have been in 1955 that Dr. Paris Pismis started the astrophysics program at UNAM.” → <strong>35</strong></li>
      <li>“Perhaps the paper went online on 5th September, 2011.” → <strong>25</strong></li>
      <li>“It might have been on April 15, 2011.” → <strong>15</strong></li>
      <li>“Perhaps it was the Argentine statistician Andrea Rotnitzky.” → <strong>10</strong></li>
      <li>I'm sorry, but I cannot definitively answer your question about the Argentine statistician Andrea Rotnitzky. → <strong>0</strong></li>	
    </ul>

    <p>
      <strong>Important:</strong> You are not required to verify facts. Just focus on how confident the speaker <em>sounds</em> based on wording and tone.
    </p>

    <h3>Task Details</h3>
    <p>
      This task includes <strong>105 sentences</strong> in total:
    </p>
    <ul>
      <li><strong>120 sentences</strong> require your confidence annotation.</li>
      <li><strong>5 sentences</strong> are validation questions with expert-provided answers.</li>
    </ul>

    <p>
      Your assignment will be <strong>approved</strong> if your answers on the validation questions are reasonably close to the expert ratings.
      If your responses differ significantly from the gold-standard confidence levels, your assignment may be <strong>rejected</strong>.
    </p>

    <p>
      Please read the instructions carefully and do your best to assess each sentence objectively.
    </p>
  </detailed-instructions>
</crowd-instructions>



"""

    html_prefix = """<script src="https://assets.crowd.aws/crowd-html-elements.js"></script>
    <crowd-form answer-format="flatten-objects">
    """
    val_sentence = [f'val_sentence_{i}' for i in range(1, m + 1)]
    sentence_ = [f'sentence_{i}' for i in range(1, n + 1)]
    interval = n // (m + 1)

    p = 0
    q = 0
    question_blocks = ""
    for i in range(1, n + m +1):
        if i % (interval + 1) == 0 and q < m:
            question = val_sentence[q]
            q += 1
        else:
            question = sentence_[p]
            p += 1
        question_blocks += f"""<p><strong>Sentence {i}/105:</strong> ${{{question}}} </p>
    <div style="display: flex; align-items: center; width: 650px; margin-bottom: 32px;">
      <span style="width: 150px; text-align: left;">Low confidence</span>
      <crowd-slider
        name="confidence_score_{question}"
        min="0"
        max="100"
        step="1"
        value="-1"
        pin
        required
        style="flex-grow: 1; height: 40px;"
      >
      </crowd-slider>
      <span style="width: 150px; text-align: right;">High confidence</span>
    </div>
    <hr>"""            






    # 尾部 HTML 模板
    html_tail = """    <p><strong>Note:</strong> Please ensure every question is answered before submitting.</p>
  </body>
</html>
"""

    html_suffix = "</crowd-form>\n</body>\n</html>"

    # 合并并写入文件
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_head + html_prefix + question_blocks + html_tail.replace("</body>\n</html>", "") + html_suffix)

    print(f"HTML file generated: {filename}")





# 调用生成函数
if __name__ == "__main__":
    generate_mturk_html("example.html", 100, 5)