import numpy as np 
import matplotlib.pyplot as plt

def format_wiki_transcript_from_convokit(corpus, convo_id, truncated_by = 0, start_at = 0): 
    """Format a wiki transcript from a convokit corpus."""
    convo = corpus.get_conversation(convo_id)
    utt_list = convo.get_chronological_utterance_list()
    transcription = []
    spk_list = {}
    if convo.meta['conversation_has_personal_attack']:
        utt_list = utt_list[:len(utt_list) - 1]
    utt_list = utt_list[:len(utt_list) - truncated_by]
    utt_list = utt_list[start_at:]
    for utt in utt_list:
        if utt.speaker.id not in spk_list.keys():
            spk_list[utt.speaker.id] = len(spk_list) + 1
        transcription.append("SPEAKER"+str(spk_list[utt.speaker.id]) +": "+utt.text)
    return transcription


def format_transcript_from_convokit(corpus, convo_id, truncated_by = 3, start_at = 0): 
    """Format a Reddit transcript from a convokit corpus."""
    convo = corpus.get_conversation(convo_id)
    utt_list = convo.get_chronological_utterance_list()
    transcription = []
    spk_list = {}
    if convo.meta['has_removed_comment']:
        utt_list = utt_list[:len(utt_list) - 1]
    utt_list = utt_list[:len(utt_list) - truncated_by]
    utt_list = utt_list[start_at:]
    for utt in utt_list:
        if utt.speaker.id not in spk_list.keys():
            spk_list[utt.speaker.id] = len(spk_list) + 1
        transcription.append("SPEAKER"+str(spk_list[utt.speaker.id]) +": "+utt.text)
    return transcription


def format_transcript_from_convokit_delta(corpus, convo_id, truncate_first_op_utt=True, truncate_last_op_utt=False): 
    """Format a Reddit delta transcript from a convokit corpus."""
    convo = corpus.get_conversation(convo_id)
    utt_list = convo.get_chronological_utterance_list()
    transcription = []
    spk_list = {utt_list[0].speaker.id : "SPEAKER1"}
    for utt in utt_list:
        if utt.speaker.id not in spk_list.keys():
            spk_list[utt.speaker.id] = "SPEAKER2"
            assert len(spk_list) == 2
        transcription.append(spk_list[utt.speaker.id] +": "+utt.text)
    if truncate_first_op_utt:
        transcription = transcription[1:]
    if truncate_last_op_utt and utt_list[-1].speaker.id == utt_list[0].speaker.id:
        transcription.pop()
    return transcription


def get_human_summary(corpus, convo_id):
    """Get the human written summary of a conversation from a convokit corpus, if it exists."""
    convo = corpus.get_conversation(convo_id)
    for summary in convo.meta['summary_meta']:
        if summary["summary_type"] == "human_written_SCD":
            return summary
    raise Exception("The conversation does not have any human written summary.")


def get_machine_summary(corpus, convo_id):
    """Get the machine generated summary of a conversation from a convokit corpus, if it exists."""
    convo = corpus.get_conversation(convo_id)
    for summary in convo.meta['summary_meta']:
        if summary["summary_type"] == "machine_generated_SCD":
            return summary
    raise Exception("The conversation does not have any human written summary.")


def get_human_summary_pair_lst(corpus):
    """Get the list of paired conversations and their human written summaries."""
    human_summary_ids = corpus.get_conversation_ids(selector=lambda conversation: conversation.meta["summary_meta"] != []
        and any(summary_meta["summary_type"] == "human_written_SCD" for summary_meta in conversation.meta["summary_meta"]))
    human_summary_pair = [] # (calm, awry) 
    for convo_id in human_summary_ids:
        convo = corpus.get_conversation(convo_id)
        if convo.meta['has_removed_comment']:
            if (convo.meta['pair_id'],convo.id) not in human_summary_pair:
                human_summary_pair.append((convo.meta['pair_id'],convo.id))
        else:
            if (convo.id, convo.meta['pair_id']) not in human_summary_pair:
                human_summary_pair.append((convo.meta['pair_id'],convo.id))
    print("Number of conversation pair: ", len(human_summary_pair))
    return human_summary_pair


def get_pair_id(corpus, convo_id):
    """Get the paired conversation's id of a conversation from a convokit corpus."""
    human_summary_pair = get_human_summary_pair_lst(corpus)
    for pair in human_summary_pair:
        if convo_id in pair:
            return pair[0] if convo_id == pair[1] else pair[1]
    raise Exception("convo not found in pairings")
    

def count_yes_no(data):
    """Count the number of yes and no judgements in a dictionary."""
    yes_count = sum(1 for item in data.values() if item['judgement'] == 'Yes')
    no_count = sum(1 for item in data.values() if item['judgement'] == 'No')
    return yes_count, no_count


def measure_score(data):
    """Measure the score of a conversation from a convokit corpus."""
    sum_score = []
    for item in data.values():
        sum_score.append(item['score'])
    return np.mean(sum_score)
    

def summarize_statistics(lst, label):
    """Summarize the statistics of a list of scores."""
    print(f"{label}")
    print(f"  Mean: {np.mean(lst):.2f}")
    print(f"  Median: {np.median(lst):.2f}")
    print(f"  25th Percentile: {np.percentile(lst, 25):.2f}")
    print(f"  75th Percentile: {np.percentile(lst, 75):.2f}")


def plot_numerical_summary(data_self, data_pair):
    """Plot the numerical summary of a list of scores."""
    summary_self = {
        'mean': np.mean(data_self),
        'median': np.median(data_self),
        'percentile_25': np.percentile(data_self, 25),
        'percentile_75': np.percentile(data_self, 75)
    }
    
    summary_pair = {
        'mean': np.mean(data_pair),
        'median': np.median(data_pair),
        'percentile_25': np.percentile(data_pair, 25),
        'percentile_75': np.percentile(data_pair, 75)
    }

    plt.figure(figsize=(12, 2))
    plt.scatter(data_self, [1] * len(data_self), color='blue', alpha=0.6, label='Self Group')
    plt.scatter(data_pair, [0] * len(data_pair), color='green', alpha=0.6, label='Pair Group')

    plt.scatter(list(summary_self.values()), [1] * 4, color='red', marker='x', s=100, label='Self Summary Stats')
    plt.scatter(list(summary_pair.values()), [0] * 4, color='orange', marker='x', s=100, label='Pair Summary Stats')

    plt.yticks([0, 1], ['Self Group', 'Pair Group'])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()


def evaluate(result):
    """Evaluate the similarity of a conversation from a convokit corpus."""
    convo_self_judgement_percent = []
    convo_pair_judgement_percent = []

    count, tied, total = 0, 0, 0

    for convo_id, convo_result in result.items():
        total += 1
        score_self = measure_score(convo_result['self'])
        score_pair = measure_score(convo_result['pair'])

        self_acc = score_self
        pair_acc = score_pair

        if self_acc > pair_acc:
            count += 1

        if self_acc < pair_acc:
            print(convo_id)

        convo_self_judgement_percent.append(self_acc)
        convo_pair_judgement_percent.append(pair_acc)

    summarize_statistics(convo_self_judgement_percent, "Evaluating Self-Simulated Conversation")
    print()
    summarize_statistics(convo_pair_judgement_percent, "Evaluating Pair-Simulated Conversation")
    plot_numerical_summary(convo_self_judgement_percent, convo_pair_judgement_percent)
    return count, tied, total, convo_self_judgement_percent, convo_pair_judgement_percent


def evaluate_two(result1, result2):
    """Evaluate the similarity of two conversations from a convokit corpus."""
    convo_self_judgement_percent = []
    convo_pair_judgement_percent = []

    count, tied, total = 0, 0, 0

    for convo_id in result1:
        total += 1
        score_self = measure_score(result1[convo_id]['self'])
        score_pair = measure_score(result1[convo_id]['pair'])

        
        score_self_mirror = measure_score(result2[convo_id]['self'])
        score_pair_mirror = measure_score(result2[convo_id]['pair'])

        self_acc = score_self + score_self_mirror
        pair_acc = score_pair + score_pair_mirror

        if self_acc > pair_acc:
            count += 1

        if self_acc < pair_acc:
            print(convo_id)

        convo_self_judgement_percent.append(self_acc)
        convo_pair_judgement_percent.append(pair_acc)

    summarize_statistics(convo_self_judgement_percent, "Evaluating Self-Simulated Conversation")
    print()
    summarize_statistics(convo_pair_judgement_percent, "Evaluating Pair-Simulated Conversation")
    plot_numerical_summary(convo_self_judgement_percent, convo_pair_judgement_percent)
    return count, tied, total, convo_self_judgement_percent, convo_pair_judgement_percent