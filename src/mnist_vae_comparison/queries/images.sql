-- Mock extract for the MNIST Digits and Fashion-MNIST VAEs.
-- Stage 01 does not execute this statement. It loads the same columns from
-- torchvision, train and test, stacked in that order.
SELECT
    image_id,
    label,
    image_split,
    dataset
FROM images;
