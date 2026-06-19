import os
from clearml import Task, OutputModel

# URI файлового сервера – берём из переменной окружения или указываем явно
FILES_URI = os.environ.get('CLEARML_FILES_HOST', 'http://192.168.0.131:8081')

tasks = Task.get_tasks(
    project_name='MLOps_course_work',
    task_filter={'status': ['completed']}
)

best_task = None
best_acc = -1.0
best_f1 = None

for task in tasks:
    if 'archived' in task.get_system_tags():
        continue
    scalars = task.get_reported_scalars()
    metrics = scalars.get('metrics', {})
    if 'accuracy' not in metrics:
        continue
    acc_values = metrics['accuracy'].get('y', [])
    if not acc_values:
        continue
    acc = acc_values[-1]
    if acc > best_acc:
        best_acc = acc
        best_task = task
        f1_values = metrics.get('f1', {}).get('y', [])
        best_f1 = f1_values[-1] if f1_values else None

if best_task is None:
    raise RuntimeError("No completed non-archived tasks with accuracy found")

print(f"Best task: {best_task.id}, accuracy: {best_acc:.4f}, f1: {best_f1:.4f}")

model_artifact = best_task.artifacts.get('model')
if not model_artifact:
    raise RuntimeError("Model artifact not found")

model_path = model_artifact.get_local_copy()

# Создаём модель с явным URI для загрузки весов на файловый сервер
output_model = OutputModel(
    task=best_task,
    name='IMDB_pipeline',
    framework='ScikitLearn',
    upload_uri=FILES_URI   # передаём строку (адрес файлового сервера)
)

output_model.set_metadata('accuracy', best_acc)
output_model.set_metadata('version', '1.0')
if best_f1 is not None:
    output_model.set_metadata('f1', best_f1)

params = best_task.get_parameters()
for p in ['max_features', 'C', 'ngram_range', 'solver', 'max_iter']:
    if p in params:
        output_model.set_metadata(p, params[p])

output_model.set_metadata('tags', 'sentiment,pipeline,logistic_regression,best_model')

# upload_uri уже задан в конструкторе, поэтому здесь не передаём
output_model.update_weights(weights_filename=model_path)
output_model.publish()

print(f"Model {output_model.id} published to Registry with accuracy {best_acc:.4f}")