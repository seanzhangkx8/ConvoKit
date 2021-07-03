from convokit.text_processing import TextProcessor,TextParser,TextToArcs
from convokit.phrasing_motifs import CensorNouns, QuestionSentences
from convokit.convokitPipeline import ConvokitPipeline

"""
Some pipelines to compute the feature representations used in each Expected Context Model demo.
"""

def parliament_arc_pipeline():
    return ConvokitPipeline([
        # to avoid most computations, we'll only run the pipeline if the desired attributes don't exist
        ('parser', TextParser(input_filter=lambda utt, aux: utt.get_info('arcs') is None)),
        ('censor_nouns', CensorNouns('parsed_censored',
                     input_filter=lambda utt, aux: utt.get_info('arcs') is None)),
        ('arcs', TextToArcs('arc_arr', input_field='parsed_censored', root_only=True,
                    input_filter=lambda utt, aux: utt.get_info('arcs') is None)),
        ('question_sentence_filter', QuestionSentences('q_arc_arr', input_field='arc_arr',
                        input_filter=lambda utt, aux: utt.get_info('q_arcs') is None)),
        ('join_arcs', TextProcessor(output_field='arcs', input_field='arc_arr', 
                   proc_fn=lambda x: '\n'.join(x), 
                    input_filter=lambda utt, aux: utt.get_info('arcs') is None)),
        ('join_q_arcs', TextProcessor(output_field='q_arcs', input_field='q_arc_arr', 
                   proc_fn=lambda x: '\n'.join(x), 
                    input_filter=lambda utt, aux: utt.get_info('q_arcs') is None))
    ])

def wiki_arc_pipeline():
    return ConvokitPipeline([
         ('parser', TextParser(input_filter=lambda utt, aux: 
                               (utt.get_info('arcs') is None)
                              and (utt.get_info('parsed') is None))),
         ('censor_nouns', CensorNouns('parsed_censored',
                     input_filter=lambda utt, aux: utt.get_info('arcs') is None)),
         ('arcs', TextToArcs('arc_arr', input_field='parsed_censored', root_only=False,
                    input_filter=lambda utt, aux: utt.get_info('arcs') is None)),
         ('join_arcs', TextProcessor(output_field='arcs', input_field='arc_arr', 
                   proc_fn=lambda x: '\n'.join(x), 
                    input_filter=lambda utt, aux: utt.get_info('arcs') is None))
    ])

def scotus_arc_pipeline():
    return ConvokitPipeline([
         ('parser', TextParser(input_filter=lambda utt, aux: utt.get_info('arcs') is None)),
         ('arcs', TextToArcs('arc_arr', input_field='parsed', root_only=False,
                    input_filter=lambda utt, aux: utt.get_info('arcs') is None)),
         ('join_arcs', TextProcessor(output_field='arcs', input_field='arc_arr', 
                   proc_fn=lambda x: '\n'.join(x), 
                    input_filter=lambda utt, aux: utt.get_info('arcs') is None))
    ])

def switchboard_text_pipeline():
    # here we don't want to overwrite alpha_text fields that already exist
    return ConvokitPipeline([
        ('text', TextProcessor(proc_fn=lambda x: x, output_field='alpha_text',
                    input_filter=lambda utt, aux: utt.get_info('alpha_text') is None))
    ])