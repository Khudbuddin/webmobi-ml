# Whisper Evaluation Report

## Model & Dataset
- **Model:** openai/whisper-small
- **Dataset:** librispeech_asr (test-clean)
- **Samples evaluated:** 20

## Summary Metrics

| Metric | Value |
|---|---|
| Word Error Rate (WER) | 18.30% |
| Character Error Rate (CER) | 5.66% |
| Average Inference Latency | 1.149s |
| Min Latency | 0.461s |
| Max Latency | 2.901s |
| Total Inference Time | 22.971s |

## What These Numbers Mean

**WER** measures the fraction of words that were wrong. A WER of 5% means 5 out of every 100 words were incorrect. Whisper-small typically achieves 3-5% WER on LibriSpeech test-clean, which is considered excellent for a small model running on CPU.

**CER** is more granular — it measures character-level errors. It is always lower than or equal to WER since partial word errors count less at the character level.

**Latency** is measured per sample. On CPU, whisper-small typically takes 2-8 seconds per clip depending on audio length. On GPU it would be under 1 second.

## Sample Predictions (first 10)

| ID | Ground Truth | Prediction |
|---|---|---|
| 0001 | MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE A | MR. QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES, AND WE ARE |
| 0002 | NOR IS MISTER QUILTER'S MANNER LESS INTERESTING THAN HIS MAT | NOR IS MR. QUILTER'S MANNER LESS INTERESTING THAN HIS MATTER |
| 0003 | HE TELLS US THAT AT THIS FESTIVE SEASON OF THE YEAR WITH CHR | HE TELLS US THAT AT THIS FESTIVE SEASON OF THE YEAR, WITH CH |
| 0004 | HE HAS GRAVE DOUBTS WHETHER SIR FREDERICK LEIGHTON'S WORK IS | HE HAS GRAVE DOUBTS WHETHER SIR FREDERICK LAYTON'S WORK IS R |
| 0005 | LINNELL'S PICTURES ARE A SORT OF UP GUARDS AND AT EM PAINTIN | LINNELL'S PICTURES ARE A SORT OF UP GUARDS AND ADAM PAINTING |
| 0006 | IT IS OBVIOUSLY UNNECESSARY FOR US TO POINT OUT HOW LUMINOUS | IT IS OBVIOUSLY UNNECESSARY FOR US TO POINT OUT HOW LUMINOUS |
| 0007 | ON THE GENERAL PRINCIPLES OF ART MISTER QUILTER WRITES WITH  | UNDER GENERAL PRINCIPLES OF ART, MR. QUILTER WRITES WITH EQU |
| 0008 | PAINTING HE TELLS US IS OF A DIFFERENT QUALITY TO MATHEMATIC | PAINTING, HE TELLS US, IS OF A DIFFERENT QUALITY TO MATHEMAT |
| 0009 | AS FOR ETCHINGS THEY ARE OF TWO KINDS BRITISH AND FOREIGN | AS FOR ETCHINGS, THEY ARE OF TWO KINDS, BRITISH AND FOREIGN. |
| 0010 | HE LAMENTS MOST BITTERLY THE DIVORCE THAT HAS BEEN MADE BETW | HE LAMENTS MOST BITTERLY THE DIVORCE THAT HAS BEEN MADE BETW |

## Limitations

- Evaluated on clean read-speech (LibriSpeech) — real-world noisy audio would likely show higher WER.
- Inference was run on CPU — latency would be ~5-10x faster on GPU.
- whisper-small is not the most accurate Whisper variant; whisper-large-v3 achieves lower WER.
- 20 samples is a small evaluation set — results may not fully represent model performance.