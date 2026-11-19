# Обучение моделей nav2ai

В `training/` перенесены тренеры FreespaceSeg, FreespaceContour и ObjectSeg,
загрузчики данных, аугментации и пакетная подготовка масок из
`autowarefoundation/autoware.off-road`, ветка `ai`, коммит
`6907554be5bea17b6304556c3804cc49e3caad0d`. Лицензия исходников Apache 2.0
находится в [model_components](../ros2_ws/src/offroad_nav2/tools/model_components/LICENSE).
Контейнер обучения и экспортёр используют одни определения моделей из
`ros2_ws/src/offroad_nav2/tools/model_components`.

## Образ и данные

Соберите образ из корня `auto-vesta`. Основа образа обучения предоставляет
совместимые PyTorch, Torchvision и CUDA. Для обучения на GPU нужен Docker с
NVIDIA Container Toolkit.

```bash
docker build -t vesta-training:local training
```

`DATA_ROOT` — абсолютный путь к внешнему каталогу с датасетами. При стандартной
подготовке каждое дерево имеет вид
`$DATA_ROOT/AORAS/<задача>/<датасет>/<датасет>/{images,gt_masks}/*.png`.
Изображение и его RGB-маска должны иметь одинаковое имя. Загрузчики выделяют
каждую десятую пару на валидацию.

Для сырых данных запустите конвертер. Например, для RELLIS-3D и FreespaceSeg:

```bash
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD:/workspace" -v "$DATA_ROOT:/data" vesta-training:local \
  python data_parsing/mask_editing.py label_config=freespaceseg_rellis3d color_map=freespaceseg
```

Вместо команды `python ...` можно передать `bash data_parsing/scripts/freespaceseg.sh`
или `bash data_parsing/scripts/objectseg.sh` для пакетной обработки. Шаблоны
входных файлов, выходные каталоги и таблицы классов лежат в `data_parsing/config/`.
FreespaceContour вычисляет контур из тех же масок, что FreespaceSeg.
Готовое отфильтрованное дерево `AORAS/FreespaceContour` можно передать через `-r`.

## Запуск обучения

```bash
mkdir -p "$HOME/.cache/torch"
train() {
  docker run --rm --gpus all --shm-size=2g --user "$(id -u):$(id -g)" \
    -e HOME=/tmp -e TORCH_HOME=/torch-cache \
    -v "$PWD:/workspace" -v "$DATA_ROOT:/data:ro" \
    -v "$HOME/.cache/torch:/torch-cache" vesta-training:local python "$@"
}

train training/FreespaceSeg/train_freespace_seg.py -r /data/AORAS/FreespaceSeg -e 100
train training/FreespaceContour/train_freespace_contour.py -r /data/AORAS/FreespaceSeg -e 100
train training/ObjectSeg/train_object_seg.py -r /data/AORAS/ObjectSeg -e 100
```

FreespaceSeg и FreespaceContour ожидают `Rellis_3D`, `Goose`, `ORFD`, `OFFSED`
и `Yamaha_CMU` внутри переданного корня. Для ObjectSeg можно выбрать набор
параметром `-d`: `Goose`, `OFFSED`, `Rellis_3D` или `CARLA_Mining`.

При старте без чекпойнта тренер использует веса EfficientNet-B0 из Torchvision.
При первом запуске они скачиваются в примонтированный кэш. `-m` загружает
`state_dict` SceneSeg. `-l -c /путь/к/весам.pth` запускает новый цикл с весов
соответствующей сети. Для продолжения того же цикла повторите команду с
`--resume`. Он читает сохранённые веса и состояние оптимизатора, а прерванную
эпоху начинает заново.

Чекпойнты по умолчанию пишутся в `training/checkpoints/<сеть>/`, логи
TensorBoard — в `training/training/runs/`. Для короткой проверки используйте
`-e 1` и уменьшенные деревья по 10 пар на датасет. Для двух Freespace-тренеров
можно уменьшить расход памяти с помощью `--micro_batch_size 2 --num_workers 2`.

## Экспорт для симулятора

Из завершённых чекпойнтов соберите отдельный снимок и экспортируйте его
штатным `make onnx` в исходном образе симулятора:

```bash
SNAPSHOT="$PWD/ml_models_snapshot"
mkdir -p "$SNAPSHOT"
cp training/checkpoints/FreespaceSeg/last_freespaceseg.pth \
   training/checkpoints/FreespaceContour/last_freespacecontour.pth \
   training/checkpoints/ObjectSeg/last_objectseg.pth "$SNAPSHOT/"
VESTA_MODELS="$SNAPSHOT" make onnx GPU=nvidia HEADLESS=1
VESTA_MODELS="$SNAPSHOT" make demo
```

Экспортёр поддерживает и плоские снимки, и вложенный формат
`<сеть>/last_*.pth`, которому отдаёт приоритет. Результат — три FP16 ONNX-файла
в `$SNAPSHOT/onnx/`: `seg.onnx`, `contour.onnx`, `objseg.onnx`. Этот каталог
доступен симулятору как `/models/onnx`.
