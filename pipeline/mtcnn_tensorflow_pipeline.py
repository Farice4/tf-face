
import random
import kfp.ds1 as ds1
import kfp.onprem as onprem
from kubernetes.client.models import V1EnvVar


platform = 'onprem'
PYTHONPATH = "/dataset/tensorflow-MTCNN"

is_aligned = False


@ds1.pipeline(
    name = 'FACE_CLASSIFICATION',
    description = 'A pipeline to train and serve the FACE CLASSFICATION example.'
)
def face_classification(train_steps='30',
                        learning_rate='-1',
                        batch_size='1000',
                        dataset_dir='/dataset',
                        output_dir='/output',
                        public_ip='10.1.0.15'):
    """
    Pipeline with three stages:
      1. prepare the mtcnn tensorflow align dataset
      1. train mtcnn tensorflow
      2. deploy atf-serving instance to the cluster
      3. deploy a web-ui to interact with it
    """
    if platform == 'onprem':
        data_vop = ds1.VolumeOp(
            name = "prepare_data_vop",
            storage_class="rook-ceph-fs",
            resource_name="mtcnn-input",
            modes=ds1.VOLUME_MODE_RWM,
            size="30Gi"
        )
        data_pvc_name = data_vop.outputs["name"]

        output_vop = ds1.VolumeOp(
            name="prepare_output_vop",
            storage_class="rook-ceph-fs",
            resource_name="mtcnn-output",
            modes=ds1.VOLUME_MODE_RWM,
            size="30Gi"
        )
        output_vop.after(data_vop)
        output_pvc_name = output_vop.outputs["name"]

    # change code and dataset to pvc
    align_dataset = ds1.ContainerOp(
        name="aligin_dataset",
        image="mtcnn-tensorflow-gpu",
        command=["/bin/sh" "-c", "echo 'begin moving data';mv /root/tensorflow-MTCNN %s/;echo 'moving is finished;"
                 % str(dataset_dir)]
    )
    align_dataset.after(output_vop)

    # train model
    train_pnet = ds1.ContainerOp(
        name='train_pnet',
        image='mtcnn-tensorflow-gpu',
        command=["/bin/sh", "-c", "cd /dataset/tensorflow-MTCNN/preprocess; python gen_12net_data.py;"
                                  "python gen_landmark_aug.py 12; python gen_imglist_pnet.py;"
                                  "python gen_tfrecords.py 12; python ../train/train.py 12"]
    ).add_resource_limit("nvidia.com/gpu", 1)
    train_pnet.after(align_dataset)

    train_rnet = ds1.ContainerOp(
        name="train_rnet",
        image="mtcnn-tensorflow-gpu",
        command=["/bin/sh", "-c", "cd /dataset/tensorflow-MTCNN/preprocess; python gen_hard_example.py 12;"
                                  "python gen_landmark_aug.py 24; python gen_tfrecords.py 24;"
                                  "python ../train/train.py 24"]
    ).add_resource_limit("nvidia.com/gpu", 1)
    train_rnet.after(train_pnet)

    train_onet = ds1.ContainerOp(
        name="train_onet",
        image="mtcnn-tensorflow-gpu",
        command=["/bin/sh", "-c", "cd /dataset/tensorflow-MTCNN/preprocess; python gen_hard_example.py 24;"
                                  "python gen_landmark_aug.py 48; python gen_tfrecords.py 48;"
                                  "python ../train/train.py 48"]
    ).add_resource_limit("nvidia.com/gpu", 1)
    train_onet.after(train_rnet)


if __name__ == '__main__':
    import kfp.compiler as compiler
    compiler.Compiler().compiler(face_classification, __file__ + '.tar.gz')
