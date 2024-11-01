# ai_detector

Изначально хотел сделать AI humanizer, но не успел
Опять же прошу не судить строго, делал за одинь день


## Datamix

Набор данных создавался поэтапно с упором на размер, разнообразие и сложность, чтобы обеспечить хорошие возможности обобщения и сильную устойчивость к состязательным примерам. На каждой итерации смешнивания данных я пыталися устранить «слепые пятна» моделей предыдущего поколения, сохраняя при этом надежность.

Чтобы максимально эффективно использовать открытые человеческие тексты, я использовал весь корпус Persuade и также включил различные человеческие тексты из таких источников, как выходной набор данных OpenAI GPT2, [ELLIPSE corpus](https://github.com/scrosseye/ELLIPSE-Corpus), NarrativeQA, wikipedia, NLTK Brown corpus.

### Источники для синтетичесих нами эссе можно сгруппировать по четырем категориям.

1. Проприетарные LLM (gpt-3.5, gpt-4, claude, cohere, gemini, palm)
2. LLM с открытым исходным кодом (Llaama, Falon, Mistral, Mixtral)
3. Существующие наборы текстовых данных, созданные LLM.
    - [Synthetic dataset made by T5](https://www.kaggle.com/datasets/conjuring92/fpe-processed-dataset?select=mlm_essays_processed.csv)
    - [DAIGT V2 subset](https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset)
    - [OUTFOX](https://github.com/ryuryukke/OUTFOX)
    - [Ghostbuster data](https://github.com/vivek3141/ghostbuster-data)
    - [gpt-2-output-dataset](https://github.com/openai/gpt-2-output-dataset)
4. Fine tune LLM с открытым исходным кодом (mistral, llama, falcon, deci-lm, t5, pythia, BLOOM, GPT2). Для Fine-Tune настройки LLM использовал [PERSUADE](https://github.com/scrosseye/persuade_corpus_2.0) разными способами:
Настройка инструкций: инструкции состояли из различных метаданных, например. подсказка имени, общий балл за эссе, статус ELL и уровень успеваемости. Ответами были соответствующие студенческие эссе.

Модели была обучена на 150k образцов (без предварительной обработки), из которых 80k были написаны человеком.

### Deberta
#### Classification: deberta-base

- Используется `AutoModelForSequenceClassification` из transformers
- `BCE loss` 

