-- Mock extract for the MNIST autoencoder.
-- Stage 01 does not execute this statement. It loads the same columns from
-- torchvision.datasets.MNIST, train and test, stacked in that order.
SELECT
    image_id,
    label,
    mnist_split
FROM images;
