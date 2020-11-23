from text_summarizer import summarizer

summarizer.text = input_text
summarizer.algo = Summ.TEXT_RANK    
summarizer.percentage = 0.25


summarizer.summarize()
summarizer.schematize()


summarizer.summarize(text_to_be_summarized)
summarizer.summarize(text_to_be_summarized, "textrank", 0.5)
