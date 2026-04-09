def detect_spikes(series, threshold=1.5):
    mean = series.mean()
    spikes = series[series > mean * threshold]
    return spikes