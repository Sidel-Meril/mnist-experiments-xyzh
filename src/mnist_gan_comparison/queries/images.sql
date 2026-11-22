-- Mock extract for the vanilla GAN.
-- Stage 01 does not execute this statement. It loads the same columns from
-- torchvision MNIST or Fashion-MNIST, train and test, stacked in that order.
-- scripts/pull_sample_data.py chooses which dataset fills this table.
SELECT
    image_id,
    label,
    mnist_split
FROM images;
