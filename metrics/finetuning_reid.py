"""Дообучение модели на ReID"""

import torchreid

from reid_dataset import ReIDDataset


def finetune():
    """Проводит дообучение модели"""

    reid_dataset = ReIDDataset
    torchreid.data.register_image_dataset('dataset1', reid_dataset)
    datamanager = torchreid.data.ImageDataManager(
        root='dataset1',
        sources=['dataset1'],
        height=256,
        width=128,
        batch_size_train=64,
        batch_size_test=128,
        transforms=['random_flip', 'random_crop']
    )
    model = torchreid.models.build_model(
        name="osnet_x1_0",
        num_classes=datamanager.num_train_pids,
        pretrained=True
    )

    optimizer = torchreid.optim.build_optimizer(
        model,
        optim="adam",
        lr=0.00001
    )

    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler="single_step",
        stepsize=20
    )

    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer,
        scheduler=scheduler,
        label_smooth=True
    )
    print('Старт обучения')
    engine.run(
        save_dir="log/custom_reid",
        max_epoch=10,
        eval_freq=1,
        print_freq=1,
        test_only=False
    )
    print('Окончание обучения')

finetune()
