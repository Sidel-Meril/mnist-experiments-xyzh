-- Mock extract for the progressive GAN.
-- Stage 01 does not execute this statement. It loads the same columns from
-- torchvision.datasets.CIFAR10, train and test, stacked in that order.
SELECT
    image_id,
    label,
    cifar_split
FROM images;
