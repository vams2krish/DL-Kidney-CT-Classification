"""Compare the bundled model with a retrained VGG16 head on one fixed image split."""
from pathlib import Path
import hashlib, json
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import tensorflow as tf

def package_model(backbone, weights, bias, path):
    layers = tf.keras.layers
    inputs = layers.Input((224, 224, 3))
    # Convert the API's [0, 1] RGB to ImageNet VGG16 mean-centered BGR.
    preprocess = layers.Conv2D(3, 1, trainable=False, name='imagenet_preprocessing')
    features = preprocess(inputs)
    preprocess.set_weights([
        (255 * np.fliplr(np.eye(3, dtype='float32'))).reshape(1, 1, 3, 3),
        np.array([-103.939, -116.779, -123.68], dtype='float32'),
    ])
    for layer in backbone.layers[1:]:
        features = layer(features)
    features = layers.Flatten()(features)
    features = layers.UnitNormalization()(features)
    head = layers.Dense(2, activation='softmax')
    outputs = head(features)
    head.set_weights([
        np.column_stack((np.zeros_like(weights), weights)).astype('float32'),
        np.array([0, bias], dtype='float32'),
    ])
    candidate = tf.keras.Model(inputs, outputs, name='kidney_vgg16_transfer')
    candidate.save(path)
    return candidate

def main():
    root = Path('artifacts/data_ingestion/kidney-ct-scan-image')
    rows = [(p, y) for y, label in enumerate(('Normal','Tumor')) for p in sorted((root / label).glob('*.jpg'))]
    images = []
    hashes = []
    for p, _ in rows:
        with Image.open(p) as im:
            rgb = im.convert('RGB')
            images.append(np.array(rgb.resize((224,224),Image.Resampling.BILINEAR)))
            hashes.append(hashlib.sha256(images[-1].tobytes()).digest())
    groups = list(range(len(rows)))
    def group(i):
        while groups[i] != i:
            i = groups[i]
        return i
    for i, a in enumerate(hashes):
        for j in range(i):
            if a == hashes[j]:
                groups[group(i)] = group(j)
    x = np.stack(images)
    y = np.asarray([label for _,label in rows])
    assert len(x)==465
    groups = [group(i) for i in range(len(x))]
    train,test = next(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(x,y,groups))
    assert len(set(y[train])) == len(set(y[test])) == 2
    old = tf.keras.models.load_model('model/model.h5',compile=False)
    backbone = tf.keras.Model(old.input,old.get_layer('block5_pool').output)
    old_preds = np.concatenate([np.asarray(old(x[idx].astype('float32')/255,training=False)).argmax(1) for idx in np.array_split(test,10)])
    features = []
    for i in range(0,len(x),16):
        f = backbone(tf.keras.applications.vgg16.preprocess_input(x[i:i+16].astype('float32')),training=False).numpy()
        features.append(f.reshape(len(f),-1))
        if i % 80 == 0: print('Features',i,flush=True)
    f = np.concatenate(features).astype('float64')
    from sklearn.preprocessing import normalize
    f = normalize(f)
    classifier = LogisticRegression(C=10,max_iter=1000).fit(f[train],y[train])
    pred = classifier.predict(f[test])
    def metrics(p):
        return dict(accuracy=accuracy_score(y[test],p),confusion_matrix=confusion_matrix(y[test],p,labels=[0,1]).tolist())
    report=dict(class_order=['Normal','Tumor'],train=len(train),test=len(test),grouping='Exact resized-pixel duplicates grouped; no patient IDs',baseline=metrics(old_preds),candidate=metrics(pred),test_files=[str(rows[i][0].relative_to(root)) for i in test],model_sha256=hashlib.sha256(Path('model/model.h5').read_bytes()).hexdigest())
    Path('.demo-logs').mkdir(exist_ok=True)
    Path('.demo-logs/retrain-results.json').write_text(json.dumps(report,indent=2))
    print({k:v for k,v in report.items() if k!='test_files'},flush=True)
    package_model(backbone, classifier.coef_[0], classifier.intercept_[0], Path('artifacts/retrained-vgg16.keras'))


if __name__ == '__main__':
    main()
