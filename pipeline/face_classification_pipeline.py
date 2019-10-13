import random
import kfp.ds1 as ds1
import kfp.onprem as onprem
from kubernetes.client.models import V1EnvVar


platform = 'onprem'
PYTHONPATH = "/dataset/face_classification/src"

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
      1. prepare the face detection align dataset
      1. train face classifier
      2. deploy atf-serving instance to the cluster
      3. deploy a web-ui to interact with it
    """
    if platform == 'onprem':
        data_vop = ds1.VolumeOp(
            name = "prepare_data_vop",
            storage_class="rook-ceph-fs",
            resource_name="classfication-input",
            modes=ds1.VOLUME_MODE_RWM,
            size="30Gi"
        )
        data_pvc_name = data_vop.outputs["name"]

        output_vop = ds1.VolumeOp(
            name="prepare_output_vop",
            storage_class="rook-ceph-fs",
            resource_name="classfication-output",
            modes=ds1.VOLUME_MODE_RWM,
            size="30Gi"
        )
        output_vop.after(data_vop)
        output_pvc_name = output_vop.outputs["name"]

    # change code and dataset to pvc
    align_dataset = ds1.ContainerOp(
        name="aligin_dataset",
        image="tensorflow-gpu/face-classification-gpu:1.3.0",
        command=["/bin/sh" "-c", "echo 'begin moving data';mv /root/face_classification %s/;echo 'moving is finished;"
                 % str(dataset_dir)]
    )
    align_dataset.after(output_vop)

    # train model
    train_emotion = ds1.ContainerOp(
        name='train_emotion',
        image='tensorflow-gpu/face-classification-gpu:1.3.0',
        command=["/bin/sh", "-c", "cd /dataset/face_classification/src; python3 train_emotion_classifier.py"]
    ).add_resource_limit("nvidia.com/gpu", 1)
    train_emotion.after(align_dataset)

    train_gender = ds1.ContainerOp(
        name="train_gender",
        image="tensorflow-gpu/face-classification-gpu:1.3.0",
        command=["/bin/sh", "-c", "cd /dataset/face_classification/src; python3 train_emotion_classifier.py"]
    ).add_resource_limit("nvidia.com/gpu", 1)
    train_gender.after(train_emotion)



if __name__ == '__main__':
    import kfp.compiler as compiler
    compiler.Compiler().compiler(face_classification, __file__ + '.tar.gz')
