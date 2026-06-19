import argparse
import glob
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from clearml import Dataset, Task, OutputModel

parser = argparse.ArgumentParser()
parser.add_argument('--max_features', type=int, default=5000)
parser.add_argument('--C', type=float, default=1.0)
parser.add_argument('--ngram_range_min', type=int, default=1)
parser.add_argument('--ngram_range_max', type=int, default=1)
parser.add_argument('--solver', type=str, default='lbfgs')
parser.add_argument('--max_iter', type=int, default=1000)
args, unknown = parser.parse_known_args()

task = Task.init(
    project_name='MLOps_course_work',
    task_name=f"IMDB_Pipeline_maxf{args.max_features}_C{args.C}_ngram{args.ngram_range_min}_{args.ngram_range_max}",
    output_uri=True
)

task.set_parameters({
    'max_features': args.max_features,
    'C': args.C,
    'ngram_range': f"({args.ngram_range_min}, {args.ngram_range_max})",
    'solver': args.solver,
    'max_iter': args.max_iter,
})

dataset = Dataset.get(
    dataset_name='IMDB Dataset',
    dataset_project='MLOps_course_work',
    dataset_version='1.0'
)
local_path = dataset.get_local_copy()
csv_files = glob.glob(f"{local_path}/*.csv")
if not csv_files:
    raise FileNotFoundError(f"CSV not found in {local_path}")
df = pd.read_csv(csv_files[0])

if 'review' not in df.columns or 'sentiment' not in df.columns:
    raise ValueError("Dataset must contain 'review' and 'sentiment' columns")

df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
if df['label'].isna().any():
    raise ValueError("sentiment column contains values other than positive/negative")

X = df['review']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(args.ngram_range_min, args.ngram_range_max),
        stop_words='english'
    )),
    ('clf', LogisticRegression(
        C=args.C,
        solver=args.solver,
        max_iter=args.max_iter,
        random_state=42
    ))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

logger = task.get_logger()
logger.report_scalar('metrics', 'accuracy', value=acc, iteration=0)
logger.report_scalar('metrics', 'f1', value=f1, iteration=0)
logger.report_single_value('accuracy', acc)
logger.report_single_value('f1', f1)

cm = confusion_matrix(y_test, y_pred)
logger.report_confusion_matrix(
    title="Confusion Matrix",
    series="test",
    iteration=0,
    matrix=cm,
    xaxis="Predicted",
    yaxis="True"
)

fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'], ax=ax)
ax.set_title('Confusion Matrix')
logger.report_matplotlib_figure(
    title="Confusion Matrix",
    series="test",
    figure=fig,
    iteration=0,
    report_image=True
)
plt.close(fig)

model_file = 'imdb_pipeline.pkl'
joblib.dump(pipeline, model_file)
task.upload_artifact('model', artifact_object=model_file) 

output_model = OutputModel(
    task=task,
    name='IMDB_pipeline',
    framework='ScikitLearn',
    tags=['sentiment', 'tfidf-logreg', f'acc={acc:.3f}', f'f1={f1:.3f}']
)

output_model.set_upload_destination('http://192.168.0.131:8081')

output_model.set_metadata('accuracy', str(acc))
output_model.set_metadata('f1', str(f1))
output_model.set_metadata('max_features', str(args.max_features))
output_model.set_metadata('C', str(args.C))
output_model.set_metadata('ngram_range', f"{args.ngram_range_min}-{args.ngram_range_max}")

output_model.update_weights(weights_filename=model_file, auto_delete_file=False)
output_model.publish()

print(f"Accuracy: {acc:.3f}, F1: {f1:.3f}")
print(f"Model registered with ID: {output_model.id}")