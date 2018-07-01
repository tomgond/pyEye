import os

FEATURE_EXTRACTOR_BINARY_PATH = "D:\\Programs\\OpenFace\\OpenFace_2.0.0_win_x64\\FeatureExtraction.exe"


def extract_features(source_file, features_file, pose=True, gaze=True)
    os.system("{binary} -f {source_file} {run_params} -of {feature_file}".format(FEATURE_EXTRACTOR_BINARY_PATH,
                                                                                 source_file,
                                                                                 "-gaze -pose",
                                                                                 features_file))

if __name__=="__main__":
    images = os.listdir("output_landmarks")
