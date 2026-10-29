'use strict';
const $ = id => document.getElementById(id);
let selected = null, synthetic = false, busy = false, previewUrl = null;
function showError(message) {
  $('errorMessage').textContent = message;
  $('errorBox').hidden = false;
  $('resultPlaceholder').hidden = true;
  $('resultCard').hidden = true;
}
function selectFile(file, isSynthetic = false) {
  if (busy) return;
  if (!['image/jpeg','image/png'].includes(file.type)) {
    selected = null; $('predictBtn').disabled = true;
    showError('Choose a JPEG or PNG image. Other formats are not supported.'); return;
  }
  if (file.size > 5 * 1024 * 1024) { selected = null; $('predictBtn').disabled = true; showError('Image exceeds the 5 MB limit.'); return; }
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
$('uploadArea').addEventListener('dragover', event => {event.preventDefault(); $('uploadArea').classList.add('dragover');});
$('uploadArea').addEventListener('dragleave', () => $('uploadArea').classList.remove('dragover'));
$('uploadArea').addEventListener('drop', event => {
  event.preventDefault(); $('uploadArea').classList.remove('dragover');
  if(event.dataTransfer.files[0]) selectFile(event.dataTransfer.files[0]);
});
$('removeBtn').addEventListener('click', () => {
  if(busy) return;
  selected = null; synthetic = false;
  if(previewUrl) URL.revokeObjectURL(previewUrl);
  $('fileInput').value = ''; $('previewContainer').hidden = true;
  $('uploadArea').hidden = false; $('predictBtn').disabled = true;
  $('resultCard').hidden = true; $('errorBox').hidden = true; $('resultPlaceholder').hidden = false;
});
$('sampleBtn').addEventListener('click', async () => {
  if(busy) return;
  try {
    const response = await fetch('/static/synthetic-demo.png');
    if(!response.ok) throw new Error('Demo image could not be loaded.');
    selectFile(new File([await response.blob()], 'synthetic-demo.png', {type:'image/png'}), true);
  } catch(error) {showError(error.message);}
});
$('dismissError').addEventListener('click', () => {$('errorBox').hidden = true; $('resultPlaceholder').hidden = false;});
$('predictBtn').addEventListener('click', async () => {
  if(!selected || busy) return;
  busy = true; $('predictBtn').disabled = true; $('removeBtn').disabled = true; $('sampleBtn').disabled = true;
  $('fileInput').disabled = true; $('resultPlaceholder').hidden = true;
  $('resultCard').hidden = true; $('errorBox').hidden = true; $('loading').hidden = false;
  const started = performance.now();
  try {
    const encoded = await new Promise((resolve, reject) => {
      const reader = new FileReader(); reader.onload = () => resolve(reader.result.split(',')[1]);
      reader.onerror = () => reject(new Error('Could not read the file.')); reader.readAsDataURL(selected);
    });
    const response = await fetch('/predict', {method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({image:encoded}), signal:AbortSignal.timeout(120000)});
    const result = await response.json();
    if(!response.ok) throw new Error(result.error || 'Classification failed.');
    const output = result[0];
    if(!output || !output.scores) throw new Error('Unexpected server response.');
    $('resultTitle').textContent = output.image;
    for(const [label, prefix] of [['Normal','normal'],['Tumor','tumor']]) {
      $(prefix+'Score').value = output.scores[label];
      $(prefix+'Percent').textContent = (output.scores[label] * 100).toFixed(2)+'%';
    }
    $('resultNote').textContent = synthetic
      ? 'Synthetic test image — this output demonstrates the software flow only. It has no medical interpretation.'
      : output.notice;
    $('latency').textContent = ((performance.now()-started)/1000).toFixed(2)+' s · completed';
    $('resultCard').hidden = false;
  } catch(error) {showError(error.name === 'TimeoutError' ? 'The model timed out. Try again or check service readiness.' : error.message);}
  finally {
    busy = false; $('loading').hidden = true; $('predictBtn').disabled = !selected;
    $('removeBtn').disabled = false; $('sampleBtn').disabled = false; $('fileInput').disabled = false;
  }
});
fetch('/health').then(response => {if(!response.ok) throw new Error(); return response.json();})
  .then(() => {$('serviceStatus').textContent = 'API online · model loads on demand';})
  .catch(() => {$('serviceStatus').textContent = 'API unavailable';});
