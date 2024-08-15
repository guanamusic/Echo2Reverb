import torch
import torchaudio.functional.functional
from scipy import signal


def normalized_echo_density(rir, segment_length=750, stride=50, n_win=106, offset=0, window_type=None):
    """
    (Description is under construction)
    Compute normalized echo density (NED) to get "normalized echo density profile".
    :param rir: (torch.Tensor) RIR waveform tensor of shape [B, L]
    :param segment_length: (int) segment length for calculating NED
    :param stride:
    :param n_win:
    :param offset:
    :param window_type:
    :return: (torch.Tensor) NED tensor of shape [B, L_padded//segment_size]
    """
    assert window_type in ['rect', 'hann'], f"window_type should be 'rect' or 'hann', not '{window_type}'"
    rir = get_rir_segments(rir, segment_length, stride, n_win, offset)

    # Windowing if windowing is true
    rir = \
        rir * torch.hann_window(segment_length).unsqueeze(0).unsqueeze(0).to(rir.device) \
        if window_type == 'hann' else rir

    # Gating by heaviside function
    gated_value = torch.heaviside(
        (torch.abs(rir) - torch.sqrt(rir.pow(2).mean(-1)).unsqueeze(-1).expand(-1, -1, rir.size(-1))),
        values=torch.Tensor([1.]).to(rir.device)
    )
    return \
        torch.linalg.norm(gated_value, dim=-1, ord=1) \
        / segment_length \
        / torch.erfc(1 / torch.sqrt(torch.tensor([2]))).item()


def differentiable_ned(rir, slope=10000, segment_length=750, stride=50, n_win=106, offset=0, window_type=None):
    """
    (Description is under construction)
    Computes differentiable approximation of normalized echo density (NED) to get normalized echo density loss.
    :param rir: (torch.Tensor) RIR waveform tensor of shape [B, L]
    :param slope: (float) slope of the sigmoid gate. Lower is looser, higher is to NP-hard
    :param segment_length: (int) segment length for calculating NED
    :param stride:
    :param n_win:
    :param offset:
    :param window_type:
    :return: (torch.Tensor) approximated NED tensor of shape [B, L_padded//segment_size]
    """
    assert window_type in ['rect', 'hann'], f"window_type should be 'rect' or 'hann', not '{window_type}'"
    rir = get_rir_segments(rir, segment_length, stride, n_win, offset)

    # Windowing if windowing is true
    rir = rir * torch.hann_window(segment_length).unsqueeze(0).unsqueeze(0).to(rir.device) \
        if window_type == 'hann' else rir

    # Normalizing
    rir = rir / (rir.pow(2).sum(-1, keepdim=True) + 1e-9).sqrt()

    # Gating by steep sigmoid function
    sigmoid_value = torch.sigmoid(
        slope * (torch.abs(rir) - torch.sqrt(rir.pow(2).mean(-1) + 1e-9).unsqueeze(-1).expand(-1, -1, rir.size(-1)))
    )
    return \
        torch.linalg.norm(sigmoid_value, dim=-1, ord=1) \
        / segment_length \
        / torch.erfc(1 / torch.sqrt(torch.tensor([2]))).item()


def get_rir_segments(rir, segment_length, stride, n_win, offset):
    """
    (Description is under construction)
    Break RIR [B, L] into [B, #_of_window, segment_length]
    """
    assert len(rir.size()) == 2, f"len(rir.size()) == {len(rir.size())}"
    pad_len = segment_length // 2
    rir = torch.nn.functional.pad(rir, (pad_len, 0), mode='constant')

    search_range = segment_length + stride * (n_win - 1)
    assert offset + search_range <= rir.size(-1), f"n_win({n_win}) is too long."

    rir = rir[:, offset:offset+search_range]

    # [B, #_of_window, segment_length]
    rir = rir.unfold(-1, segment_length, stride)
    assert rir.size(1) == n_win, rir.size()

    return rir


def get_filterbank(filterbank_type, sample_rate=48000, signal_length=3000, synthesis=True):
    """
    (Description is under construction)
    Create a mel filterbank for synthesize new noise segment.
    :param filterbank_type:
    :param sample_rate: (int)
    :param signal_length: (int)
    :param synthesis: whether synthesis filterbank or analysis filterbank
    :return: mel filterbank in freq domain of shape [signal_length, n_filter]
    """
    assert signal_length % 2 == 0, f"'signal_length' must be even, not {signal_length}"
    assert filterbank_type in ["mel", "octave", "1/3-octave", "merged_octave"], \
        f"filterbank_type: '{filterbank_type}' is not yet supported."

    all_freqs = torch.linspace(0, sample_rate // 2, signal_length // 2 + 1)
    if filterbank_type == "mel":
        n_mels = 128
        m_min = torchaudio.functional.functional._hz_to_mel(0)
        m_max = torchaudio.functional.functional._hz_to_mel(sample_rate // 2)
        m_pts = torch.linspace(m_min, m_max, n_mels + 2)
        f_pts = torchaudio.functional.functional._mel_to_hz(m_pts)
    elif filterbank_type in ["octave", "merged_octave"]:
        center_freqs = [16, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        f_pts = torch.tensor([0] + center_freqs + [sample_rate // 2])
    elif filterbank_type == "1/3-octave":
        center_freqs = \
            [16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500] \
            + [630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000]
        f_pts = torch.tensor([0] + center_freqs + [sample_rate // 2])
    else:
        raise NotImplementedError

    fb = torchaudio.functional.functional._create_triangular_filterbank(all_freqs, f_pts)
    if filterbank_type == "merged_octave":
        fb = torch.cat((fb[..., 0:4].sum(-1, keepdim=True), fb[..., 4:]), dim=-1)

    # Building whole filterbank from one-sided filterbank
    filterbank = torch.cat((fb, torch.flip(fb[1:-1], dims=[0])), dim=0).detach() if synthesis else fb.detach()

    return filterbank


def velvet_noise_generation(average_pulse_distance, interleave_position, pulse_log_gain, shape_tuple=(16, 40, 4, 750)):
    """
    (Description is under construction)
    Generate the velvet noise.
    Velvet noises with different sparsity for a fixed length are simultaneously generated.
    :param average_pulse_distance: (int) sparsity (T_d)
    :param interleave_position:
    :param pulse_log_gain: pulse_log_gain definitely
    :param shape_tuple: (tuple) (B, C1, C2, L)
    :return: (torch.Tensor) velvet noise of shape [(shape_tuple)]
    """
    assert average_pulse_distance == 750//3
    gate_fn = lambda x, bias, : \
        1 - torch.heaviside(x - bias, values=torch.Tensor([1.]).to(pulse_log_gain.device)).detach()

    sparsity = torch.ones(shape_tuple[:-1] + (1,)).to(pulse_log_gain.device) * average_pulse_distance
    signal_length = int(shape_tuple[-1])
    max_pulse_num = int(signal_length / average_pulse_distance)

    # Generating the impulse locations, [..., max_pulse_num]
    uniform_vec = torch.rand(shape_tuple[:-1] + (max_pulse_num,)).to(pulse_log_gain.device)
    impulse_location_float = \
        uniform_vec * (sparsity - 1 - interleave_position) \
        + torch.ones_like(uniform_vec) * torch.arange(max_pulse_num).to(pulse_log_gain.device) * sparsity
    impulse_location_int = torch.round(impulse_location_float)
    impulse_location = impulse_location_int * gate_fn(impulse_location_int, bias=signal_length)
    # Compensating zero-index-pulse
    zero_count = (max_pulse_num - 1) - torch.count_nonzero(impulse_location[..., 1:], dim=-1)
    velvet = torch.sum(
        torch.nn.functional.one_hot(impulse_location.type(torch.long), num_classes=signal_length),
        dim=-2
    )
    velvet[..., 0] = velvet[..., 0] - zero_count

    # NOTE: Do not multiplying binary amplitude {-1, +1}!
    velvet = velvet.view(velvet.size()[:-1] + (-1, average_pulse_distance))
    velvet = velvet * torch.exp(pulse_log_gain).unsqueeze(-1)
    return velvet.view(shape_tuple)


def hpf_before_get_echo_density(rir, e2r_trigger=False):
    assert len(rir.size()) == 2

    spectral_weight = torch.ones(128)
    spectral_weight[:20] = 0.

    fb = get_filterbank(filterbank_type='mel', sample_rate=48000, signal_length=3000, synthesis=True)
    spectral_weight_fourier = (fb.transpose(-1, -2) * spectral_weight.unsqueeze(-1)).sum(-2)
    window = torch.from_numpy(signal.windows.hann(3000)).to(torch.float32)

    zero_pad_len_front = 2250 if not e2r_trigger else 2200
    zero_pad_len_rear = 2250
    rir = torch.nn.functional.pad(rir, (zero_pad_len_front, zero_pad_len_rear), mode='constant')
    rir = rir.unfold(-1, 3000, 750) * window.to(rir.device)
    filtered_rir = torch.fft.ifft(
        torch.fft.fft(rir, dim=-1, n=3000) * spectral_weight_fourier.to(rir.device),
        dim=-1,
        n=3000
    ).real
    filtered_rir = torch.nn.functional.fold(
        input=filtered_rir.transpose(-1, -2),
        output_size=(1, 120000 + 2250 * 2 if not e2r_trigger else 3750 + 2250 * 2),
        kernel_size=(1, 3000),
        stride=750
    ).squeeze(-2)
    return filtered_rir[..., zero_pad_len_front:-zero_pad_len_rear] / 2
