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
    m1 = re.match(r"^(\D\d)([0-9]+)$", text, re.I)
    m2 = re.match(r"^(\d\D)([0-9]+)$", text, re.I)
    m3 = re.match(r"^([0-9]{3,4})([a-z]{2,3})$", text, re.I)
    m4 = re.match(r"^([a-z]{2,3})([0-9]{3,4})$", text, re.I)
    m5 = re.match(r"^([0-9]{3,4})(\d\D)$", text, re.I)
    m6 = re.match(r"^([0-9]{3,4})(\D\d)$", text, re.I)

    for m_ in [m1, m2]:
        if m_:
            return '-'.join(m_.groups()) if len(m_.groups()[1]) < 5 else ''

    m = [m3, m4, m5, m6]
    for m_ in m:
        if m_:
            return '-'.join(m_.groups())

    m7 = re.match(r"^((\d)\2)(\d{4})$", text)
    m8 = re.match(r"^(\d{4})((\d)\3)$", text)
    if m7 and m8:
        return text
    if m7:
        return '-'.join(m7.groups()[0::2])
    elif m8:
        return '-'.join(m8.groups()[0:2])
    return ''
