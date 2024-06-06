import re
import numpy as np


# ctc decode
NINF = -1 * float('inf')
DEFAULT_EMISSION_THRESHOLD = 0.01

def _reconstruct(labels, blank=0):
    new_labels = []
    # merge same labels
    previous = None
    for l in labels:
        if l != previous:
            new_labels.append(l)
            previous = l
    # delete blank
    new_labels = [l for l in new_labels if l != blank]

    return new_labels


def beam_search_decode(emission_log_prob, blank=0, **kwargs):
    beam_size = kwargs['beam_size']
    emission_threshold = kwargs.get('emission_threshold', np.log(DEFAULT_EMISSION_THRESHOLD))

    length, class_count = emission_log_prob.shape

    beams = [([], 0)]  # (prefix, accumulated_log_prob)
    for t in range(length):
        new_beams = []
        for prefix, accumulated_log_prob in beams:
            for c in range(class_count):
                log_prob = emission_log_prob[t, c]
                if log_prob < emission_threshold:
                    continue
                new_prefix = prefix + [c]
                # log(p1 * p2) = log_p1 + log_p2
                new_accu_log_prob = accumulated_log_prob + log_prob
                new_beams.append((new_prefix, new_accu_log_prob))

        # sorted by accumulated_log_prob
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]

    # sum up beams to produce labels
    total_accu_log_prob = {}
    for prefix, accu_log_prob in beams:
        labels = tuple(_reconstruct(prefix))
        # log(p1 + p2) = logsumexp([log_p1, log_p2])
        _logsumexp = np.log(np.sum(np.exp([accu_log_prob, total_accu_log_prob.get(labels, NINF)])))
        total_accu_log_prob[labels] = _logsumexp

    labels_beams = [(list(labels), accu_log_prob)
                    for labels, accu_log_prob in total_accu_log_prob.items()]
    labels_beams.sort(key=lambda x: x[1], reverse=True)
    labels = labels_beams[0][0]
    return labels


def ctc_decode(log_probs, label2char=None, blank=0, beam_size=10):
    emission_log_probs = np.transpose(log_probs, (1, 0, 2))
    # size of emission_log_probs: (batch, length, class)

    decoder = beam_search_decode

    decoded_list = []
    for emission_log_prob in emission_log_probs:
        decoded = decoder(emission_log_prob, blank=blank, beam_size=beam_size)
        if label2char:
            decoded = [label2char[l] for l in decoded]
        decoded_list.append(decoded)
    return decoded_list


def text_add_dash(text):
    rules = [
        r"^(\D\d)([0-9]{2,4})$",  # A1-01, A1-001, A1-0001
        r"^(\d\D)([0-9]{2,4})$",  # 1A-01, 1A-001, 1A-0001
        r"^([0-9]{3,4})([a-z]{2})$",  # 001-AA, 0001-AA
        r"^([0-9]{3})([a-z]{3})$",  # 001-AAA
        r"^([0-9]{2})([a-z]{2})$",  # 01-AA
        r"^([a-z]{2})([0-9]{2,4})$",  # AA-01, AA-001, AA-0001
        r"^([a-z]{3})([0-9]{3,4})$",  # AAA-001, AAA-0001
        r"^([0-9]{3,4})(\d\D)$",  # 001-1A, 0001-1A
        r"^([0-9]{3,4})(\D\d)$",  # 001-A1, 0001-A1
        r"^([0-9]{1}\D{2})(\d{3})$",  # 1AA-001
        r"^(\d{3})(\d{4})$",  # 001-0001 [1984/11-1992]
        r"^(\d{2})(\d{4})$",  # 01-0001 [1979-1984]
        r"^(\d{3})(\d{3})$",  # 001-001 [1981]

    ]

    for rule in rules:
        m = re.match(rule, text, re.I)
        if m:
            return '-'.join(m.groups())

    # 微型電動二輪車
    if re.match(r"^(\D{2})(\d{5})$", text, re.I):
        return text
    else:
        return ''
