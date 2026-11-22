-- Mock extract for the MNIST RealNVP demo.
-- Stage 01 does not execute this statement. It loads the same columns from
-- torchvision MNIST and FashionMNIST, train and test, stacked in that order.
SELECT
    image_id,
    label,
    mnist_split,
    dataset,
    label_name
FROM images;
