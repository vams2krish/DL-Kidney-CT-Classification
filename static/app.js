'use strict';
const $ = id => document.getElementById(id);
let selected = null, synthetic = false, busy = false, previewUrl = null;
const modelMode = () => document.querySelector('input[name="modelMode"]:checked').value;

function updateModelSelection() {
  const fourClass = modelMode() === 'four_class';
  $('activeModel').textContent = fourClass ? 'Four-class CNN' : 'VGG16';
  $('modelSpecs').textContent = fourClass
    ? 'Four classes · 64 × 64 grayscale'
    : 'Binary classification · 224 × 224 RGB';
  $('resultCard').hidden = true;
  $('errorBox').hidden = true;
  $('resultPlaceholder').hidden = false;
}
$('modelPicker').addEventListener('change', updateModelSelection);

function renderScores(scores) {
  const rows = Object.entries(scores).map(([label, score]) => {
    const row = document.createElement('div');
    row.className = 'score-row';
    const name = document.createElement('span');
    name.textContent = label;
    const bar = document.createElement('progress');
    bar.max = 1;
    bar.value = score;
    bar.setAttribute('aria-label', label + ' raw model score');
    const percentage = document.createElement('strong');
    percentage.textContent = (score * 100).toFixed(2) + '%';
    row.append(name, bar, percentage);
    return row;
  });
  $('scoreList').replaceChildren(...rows);
}
function showError(message) {
  $('errorMessage').textContent = message;
  $('errorBox').hidden = false;
  $('resultPlaceholder').hidden = true;
  $('resultCard').hidden = true;
}
function clearSelection() {
  selected = null;
  synthetic = false;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  $('previewImage').removeAttribute('src');
  $('fileInput').value = '';
  $('previewContainer').hidden = true;
  $('uploadArea').hidden = false;
  $('predictBtn').disabled = true;
}
function selectFile(file, isSynthetic = false) {
  if (busy) return;
  if (!['image/jpeg','image/png'].includes(file.type)) {
    clearSelection();
    showError('Choose a JPEG or PNG image. Other formats are not supported.'); return;
  }
  if (file.size > 5 * 1024 * 1024) {
    clearSelection();
    showError('Image exceeds the 5 MB limit.');
    return;
  }
  selected = file; synthetic = isSynthetic;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  $('previewImage').src = previewUrl;
  $('fileName').textContent = file.name;
  $('imageTag').textContent = synthetic ? 'SYNTHETIC · NOT A PATIENT SCAN' : 'YOUR IMAGE';
  $('previewContainer').hidden = false; $('uploadArea').hidden = true;
  $('predictBtn').disabled = false; $('errorBox').hidden = true;
  $('resultCard').hidden = true; $('resultPlaceholder').hidden = false;
}
$('fileInput').addEventListener('change', event => {
  if (event.target.files[0]) selectFile(event.target.files[0]);
});
$('replaceBtn').addEventListener('click', () => {
  $('fileInput').value = '';
  $('fileInput').click();
});
$('uploadArea').addEventListener('dragover', event => {event.preventDefault(); $('uploadArea').classList.add('dragover');});
$('uploadArea').addEventListener('dragleave', () => $('uploadArea').classList.remove('dragover'));
$('uploadArea').addEventListener('drop', event => {
  event.preventDefault(); $('uploadArea').classList.remove('dragover');
  if(event.dataTransfer.files[0]) selectFile(event.dataTransfer.files[0]);
});
$('removeBtn').addEventListener('click', () => {
  if(busy) return;
  clearSelection();
  $('resultCard').hidden = true; $('errorBox').hidden = true; $('resultPlaceholder').hidden = false;
  $('fileInput').focus();
});
$('sampleBtn').addEventListener('click', async () => {
  if(busy) return;
  try {
    const response = await fetch('/static/synthetic-demo.png');
    if(!response.ok) throw new Error('Demo image could not be loaded.');
    selectFile(new File([await response.blob()], 'synthetic-demo.png', {type:'image/png'}), true);
  } catch(error) {showError(error.message);}
});
$('dismissError').addEventListener('click', () => {
  clearSelection();
  $('errorBox').hidden = true;
  $('resultPlaceholder').hidden = false;
  $('fileInput').click();
});
$('predictBtn').addEventListener('click', async () => {
  if(!selected || busy) return;
  busy = true; $('predictBtn').disabled = true; $('removeBtn').disabled = true; $('sampleBtn').disabled = true;
  $('replaceBtn').disabled = true;
  $('modelPicker').disabled = true;
  $('resultsPanel').setAttribute('aria-busy', 'true');
  $('predictLabel').textContent = 'Classifying…';
  $('fileInput').disabled = true; $('resultPlaceholder').hidden = true;
  $('resultCard').hidden = true; $('errorBox').hidden = true; $('loading').hidden = false;
  const started = performance.now();
  try {
    const encoded = await new Promise((resolve, reject) => {
      const reader = new FileReader(); reader.onload = () => resolve(reader.result.split(',')[1]);
      reader.onerror = () => reject(new Error('Could not read the file.')); reader.readAsDataURL(selected);
    });
    const response = await fetch('/predict', {method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({image:encoded, mode:modelMode(), explain:true}), signal:AbortSignal.timeout(120000)});
    const result = await response.json();
    if(!response.ok) throw new Error(result.error || 'Classification failed.');
    const output = result[0];
    if(!output || !output.scores) throw new Error('Unexpected server response.');
    $('resultTitle').textContent = output.image;
    $('scanDescription').textContent = output.description || '';
    $('attentionPanel').hidden = !output.attention_image;
    if (output.attention_image) {
      $('attentionImage').src = 'data:image/png;base64,' + output.attention_image;
    } else {
      $('attentionImage').removeAttribute('src');
    }
    renderScores(output.scores);
    $('resultModel').textContent = output.model;
    $('resultNote').textContent = synthetic
      ? 'Synthetic test image — this output demonstrates the software flow only. It has no medical interpretation.'
      : output.notice;
    $('latency').textContent = ((performance.now()-started)/1000).toFixed(2)+' s · completed';
    $('resultCard').hidden = false;
    $('resultsPanel').setAttribute('aria-busy', 'false');
    $('resultTitle').focus();
  } catch(error) {showError(error.name === 'TimeoutError' ? 'The model timed out. Try again or check service readiness.' : error.message);}
  finally {
    busy = false; $('loading').hidden = true; $('predictBtn').disabled = !selected;
    $('predictLabel').textContent = 'Run classification';
    $('replaceBtn').disabled = false;
    $('modelPicker').disabled = false;
    $('resultsPanel').setAttribute('aria-busy', 'false');
    $('removeBtn').disabled = false; $('sampleBtn').disabled = false; $('fileInput').disabled = false;
  }
});
fetch('/health').then(response => {if(!response.ok) throw new Error(); return response.json();})
  .then(() => {$('serviceStatus').textContent = 'API online · model loads on demand';})
  .catch(() => {$('serviceStatus').textContent = 'API unavailable';});
