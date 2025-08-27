---
layout: default
title: "FR3-D: A Regressor-Guided SE(3)-Equivariant conditioned Diffusion Model for 3D Fracture Reassembly"
---
<div style="text-align:center;">
    <div style="font-size: 1.3em;"><strong>Cederic Aßmann</strong></div>
    <div style="font-size: 1.1em;">Technical University Berlin, Learning and Intelligent Systems Lab</div>
    <div>
        <a href="https://www.linkedin.com/in/cederic-aßmann-41904322b" target="_blank">
            <img src="{{ '/assets/icons/li.png' | relative_url }}" width="150" alt="LinkedIn">
        </a>
        &nbsp;&nbsp;
        <a href="https://github.com/cederican/FR3-D" target="_blank">
            <img src="{{ '/assets/icons/gi.png' | relative_url }}" width="150" alt="GitHub">
        </a>
    </div>
</div>
<br>

<script type="text/javascript"
  id="MathJax-script"
  async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>


<div style="text-align:center;">
  <h1>Abstract</h1>
</div>

<div style="text-align:justify; max-width:900px; margin:auto;">

The problem of 3D fracture reassembly arises in diverse domains such as medicine, cultural heritage preservation, and archaeology. While recent data-driven approaches have demonstrated promising results on synthetically fractured objects, they still fall short of providing robust and fully generalisable solutions. In this thesis, we introduce <span style="color:#00CBFF;">FR3-D</span>: a Regressor-Guided SE(3)-Equivariant Conditioned Diffusion Model for 3D Fracture Reassembly. <span style="color:#00CBFF;">FR3-D</span> builds on a pretrained VQ-PointNet++ latent feature encoder and integrates additional fracture surface scores and surface normal information, enabling the diffusion-based denoising process to reason more effectively about geometric consistency between fragments. Conditioning on SE(3)-equivariant representations proves particularly influential with high compression rates of the encoder. Furthermore, we propose a quality estimation regressor, supervised by ground-truth reassembly metrics, which guides the probabilistic sampling process. This regressor-based extension allows <span style="color:#00CBFF;">FR3-D</span> to harness the full potential of the diffusion framework and improves reliability in selecting plausible reassembly candidates. Empirical evaluation demonstrates the effectiveness of our approach on the synthetic Breaking Bad benchmark and a novel fractured tibia bone dataset, <span style="color:#00CBFF;">FR3-D</span> achieves 85.9% lower translation error and 75.6% lower rotation error on the artifact subset compared to PuzzleFusion++, while also surpassing GARF-mini on the everyday object subset through the proposed reassembly quality-driven sampling strategy. Importantly, we show that models trained on synthetic fractures can plausibly reconstruct real fractured bones, highlighting the transferability of our framework beyond controlled synthetic settings.

</div>
<br>

<div style="text-align:center">
  <img src="{{ '/fig/teaser.gif' | relative_url }}" style="width:100%; max-width:900px;" alt="Cool GIF">
</div>

<br>

<div style="text-align:center;">
  <h1>FR3-D Overview</h1>
</div>

<div style="text-align:center">
  <img src="{{ '/fig/arch.png' | relative_url }}" style="width:100%; max-width:900px;" alt="Cool Arch">
</div>

<div style="text-align:justify; max-width:900px; margin:auto;">

<span style="color:black;">(a)</span> The pretrained VQ-PointNet++ Encoder is used to provide additional input features besides the noisy translation and rotation vector \(\mathbf{x}_t\) to the denoiser. SE(3)-Equivariant Attention module can be used as further conditioning of the denoiser. The final pose prediction \(\mathbf{x}_0\) is acquired according to the noise scheduler settings. <span style="color:black;">(b)</span> The reassembly quality estimation regressor ranks the quality of pose predictions from the denoiser supervised by combined ground truth metric scalar scores \(\xi_m\).

</div>
<br>

<div style="text-align:center;">
  <h1>3D Fracture Reassembly Results</h1>
</div>

<div style="text-align:justify; max-width:900px; margin:auto;">

We evaluate <span style="color:#00CBFF;">FR3-D</span> qualitatively on the baseline Breaking Bad Everyday and Artifact subsets, the Synthetic Fractured Tibia Bones and the Real Fractured Tibia Bones.  

</div>
<br>

<div style="text-align:center;">
  <h2>Breaking Bad Dataset</h2>
</div>

<div style="display:flex; justify-content:center; flex-wrap:wrap; gap:10px; margin-bottom:20px;">
  <video autoplay loop muted playsinline style="width:22%;" onloadedmetadata="this.playbackRate=1.5;">
    <source src="{{ '/assets/video/everyday/260_acc0.857/video.mp4' | relative_url }}" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <video autoplay loop muted playsinline style="width:22%;" onloadedmetadata="this.playbackRate=1.5;">
    <source src="{{ '/assets/video/everyday/839_acc1.0/video.mp4' | relative_url }}" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <video autoplay loop muted playsinline style="width:22%;" onloadedmetadata="this.playbackRate=1.5;">
    <source src="{{ '/assets/video/everyday/2253_acc1.0/video.mp4' | relative_url }}" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <video autoplay loop muted playsinline style="width:22%;" onloadedmetadata="this.playbackRate=1.5;">
    <source src="{{ '/assets/video/everyday/6074_acc1.0/video.mp4' | relative_url }}" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>

<div style="text-align:center;">
  Everyday Dataset
</div>
<br>

<div style="display:flex; justify-content:center; flex-wrap:wrap; gap:10px; margin-bottom:20px;">
  <video autoplay loop muted playsinline style="width:22%;" onloadedmetadata="this.playbackRate=1.5;">
    <source src="{{ '/assets/video/artifact/102_acc1.0/video.mp4' | relative_url }}" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <video autoplay loop muted playsinline style="width:22%;" onloadedmetadata="this.playbackRate=1.5;">
    <source src="{{ '/assets/video/artifact/456_acc0.875/video.mp4' | relative_url }}" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <video autoplay loop muted playsinline style="width:22%;" onloadedmetadata="this.playbackRate=1.5;">
    <source src="{{ '/assets/video/artifact/571_acc0.933/video.mp4' | relative_url }}" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <video autoplay loop muted playsinline style="width:22%;" onloadedmetadata="this.playbackRate=1.5;">
    <source src="{{ '/assets/video/artifact/3035_acc0.95/video.mp4' | relative_url }}" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>

<div style="text-align:center;">
  Artifact Dataset
</div>
<br>


<div style="text-align:center;">
  <h2>Synthetic Fractured Bones</h2>
</div>

<div style="display:flex; justify-content:center; flex-wrap:wrap; gap:10px; margin-bottom:20px;">
  <video autoplay loop muted playsinline style="width:22%;" onloadedmetadata="this.playbackRate=1.5;">
    <source src="{{ '/assets/video/synthetic/952_acc1.0/video.mp4' | relative_url }}" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <video autoplay loop muted playsinline style="width:22%;" onloadedmetadata="this.playbackRate=1.5;">
    <source src="{{ '/assets/video/synthetic/1003_acc0.8/video.mp4' | relative_url }}" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <video autoplay loop muted playsinline style="width:22%;" onloadedmetadata="this.playbackRate=1.5;">
    <source src="{{ '/assets/video/synthetic/1378_acc0.947/video.mp4' | relative_url }}" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <video autoplay loop muted playsinline style="width:22%;" onloadedmetadata="this.playbackRate=1.5;">
    <source src="{{ '/assets/video/synthetic/1408_acc1.0/video.mp4' | relative_url }}" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>

<div style="text-align:center;">
  Advanced Tibia Bone Dataset
</div>
<br>

<div style="text-align:center;">
  <h2>Real Fractured Bones</h2>
</div>

<div style="display:flex; justify-content:center; flex-wrap:wrap; gap:10px; margin-bottom:20px;">
  <video autoplay loop muted playsinline style="width:22%;" onloadedmetadata="this.playbackRate=1.5;">
    <source src="{{ '/assets/video/real/1_acc0.222/video.mp4' | relative_url }}" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <video autoplay loop muted playsinline style="width:22%;" onloadedmetadata="this.playbackRate=1.5;">
    <source src="{{ '/assets/video/real/4_acc0.2/video.mp4' | relative_url }}" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <video autoplay loop muted playsinline style="width:22%;" onloadedmetadata="this.playbackRate=1.5;">
    <source src="{{ '/assets/video/real/7_acc0.375/video.mp4' | relative_url }}" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <video autoplay loop muted playsinline style="width:22%;" onloadedmetadata="this.playbackRate=1.5;">
    <source src="{{ '/assets/video/real/9_acc0.111/video.mp4' | relative_url }}" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>

<div style="text-align:center;">
  Real Tibia Bone Dataset
</div>
<br>
<br>

<div style="text-align:center;">
  <h2>Acknowledgements</h2>
</div>

<div style="text-align:justify; max-width:900px; margin:auto;">

The <span style="color:#00CBFF;">FR3-D</span> framework is deeply inspired by PuzzleFusion++ and leverages ideas from Jigsaw and GARF. We benefited from their open-source code. Please consider reading these papers if interested in relevant topics. We gratefully acknowledge the Charité Universitätsmedizin for access to curated healthy and fractured CT Tibia Bones.
</div>

