# feature encoding functions
def prepare_encoder(train_samples, features):
    features_value_encoder = [{} for _ in features.keys()]

    for i in range(len(features_value_encoder)):
        encode_value = 0
        _field_value_encoder = features_value_encoder[i]

        for sample in train_samples.itertuples(index=False):
            field_value = sample[i]
            if field_value in _field_value_encoder:
                continue
            else:
                _field_value_encoder[field_value] = encode_value
                encode_value += 1

    # set unseen feature value's encode value
    for feature_value_encoder in features_value_encoder:
        unseen_value = max(feature_value_encoder.values()) + 1
        feature_value_encoder["unseen"] = unseen_value

    return features_value_encoder

def encode_train_samples(train_samples, encoder):
    encoded_train_samples = []
    for sample in train_samples.itertuples(index=False):
        _sample = []
        for i in range(len(sample)-1):
            encoded_value = encoder[i][sample[i]]
            _sample.append(encoded_value)
        encoded_train_samples.append(_sample)

    return encoded_train_samples

def encode_test_samples(test_samples, encoder):
    encoded_test_samples = []
    for sample in test_samples.itertuples(index=False):
        _sample = []
        for i in range(len(sample)):
            feature_encoder = encoder[i]
            try:
                encoded_value = feature_encoder[sample[i]]
            except KeyError:
                # encode feature value not seen in train set as max feature encode value + 1
                print("feature {} value {} not seen in training set".format(i, sample[i]))
                encoded_value = feature_encoder["unseen"]
            _sample.append(encoded_value)
        encoded_test_samples.append(_sample)

    return encoded_test_samples
