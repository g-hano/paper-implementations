I followed [that](https://youtu.be/vAmKB7iPkWw?si=G3uTQroppYqyyv5N) tutorial, all code belongs to [hkproj](https://github.com/hkproj)

**Follow code in this order:**
1. [modeling_siglip.ipynb](https://github.com/g-hano/paper-implementations/blob/main/PaliGemma/modeling_siglip.ipynb)
    - In this notebook we code Siglip part of the PaliGemma.
      ![siglip.png](PaliGemma/imgs/siglip.png)
    
2. [modeling_paligemma.ipynb](https://github.com/g-hano/paper-implementations/blob/main/PaliGemma/processing_paligemma.ipynb)
    - In this notebook, we define PaliGemma Processor. It is used for concatinating the vision and text embeddings.
      ![paligemma.png](PaliGemma/imgs/paligemma.png)
    
3. [modeling_gemma.ipynb](https://github.com/g-hano/paper-implementations/blob/main/PaliGemma/modeling_gemma.ipynb)
    - In final code, we code Gemma. This part is the longest and most complex one.
      ![gemma.png](PaliGemma/imgs/gemma.png)
