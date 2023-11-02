from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def svm_pipeline(kernel='rbf'):
    pipeline = make_pipeline(
        StandardScaler(),
        SVC(kernel=kernel)
    )

    return pipeline