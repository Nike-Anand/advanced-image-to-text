import sys
import os
sys.path.append('..')

from ocr_tamil.ocr import OCR as BaseOCR
from charset_utils import get_charset_for_languages, LANGUAGE_CHARSETS
from ocr_tamil.strhub.data.utils import Tokenizer
import torch

class MultiLangOCR(BaseOCR):
    """Extended OCR class with multi-language support"""
    
    def __init__(self, detect=False, lang=["tamil", "english"], multilang_model_path=None, **kwargs):
        # Store multilang model path
        self.multilang_model_path = multilang_model_path
        self.supported_languages = lang
        
        # Initialize base class
        super().__init__(detect=detect, lang=lang, **kwargs)
    
    def load_model(self):
        """Override to load multi-language models"""
        self.img_transform = self.get_transform()
        
        # Create charset for supported languages
        self.multilang_charset = get_charset_for_languages(self.supported_languages)
        self.multilang_tokenizer = Tokenizer(self.multilang_charset)
        
        # Load English model (existing)
        self.eng_character_set = """0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""
        self.eng_tokenizer = Tokenizer(self.eng_character_set)
        
        # Import PARSeq class
        from ocr_tamil.strhub.models.parseq.system import PARSeq
        import torch.serialization
        torch.serialization.add_safe_globals([('ocr_tamil.strhub.models.parseq.system', 'PARSeq')])
        
        # Load models based on precision
        if self.fp16:
            self.eng_parseq = self._load_english_model().to(self.device).half().eval()
            self.multilang_parseq = self._load_multilang_model().to(self.device).half().eval()
        else:
            self.eng_parseq = self._load_english_model().to(self.device).eval()
            self.multilang_parseq = self._load_multilang_model().to(self.device).eval()
    
    def _load_english_model(self):
        """Load English model"""
        from ocr_tamil.strhub.models.utils import load_from_checkpoint
        return load_from_checkpoint("pretrained=parseq")
    
    def _load_multilang_model(self):
        """Load multi-language model"""
        if self.multilang_model_path and os.path.exists(self.multilang_model_path):
            return torch.load(self.multilang_model_path, weights_only=False)
        else:
            # Fallback to Tamil model if multilang not available
            return torch.load(self.tamil_model_path, weights_only=False)
    
    def detect_language(self, text_confidence_pairs):
        """Detect most likely language based on character patterns"""
        language_scores = {lang: 0 for lang in self.supported_languages}
        
        for text, conf in text_confidence_pairs:
            for char in text:
                for lang, charset in LANGUAGE_CHARSETS.items():
                    if char in charset and lang in self.supported_languages:
                        language_scores[lang] += conf
        
        # Return language with highest score
        if any(language_scores.values()):
            return max(language_scores, key=language_scores.get)
        return "english"  # Default fallback
    
    def text_recognize_batch(self, exported_regions):
        """Enhanced text recognition with multi-language support"""
        dataset = self.ParseqDataset(exported_regions, transform=self.img_transform)
        dataloader = self.DataLoader(dataset, batch_size=self.batch_size)
        
        multilang_label_list = []
        multilang_confidence_list = []
        eng_label_list = []
        eng_confidence_list = []
        
        for data in dataloader:
            if self.fp16:
                data = data.to(self.device).half()
            else:
                data = data.to(self.device)
            
            # Multi-language prediction
            if len(self.supported_languages) > 1:
                with torch.cuda.amp.autocast() and torch.inference_mode():
                    logits = self.multilang_parseq(data)
                pred = logits.softmax(-1)
                label, confidence = self.multilang_tokenizer.decode(pred)
                multilang_label_list.extend(label)
                multilang_confidence_list.extend(confidence)
            else:
                multilang_label_list.extend(["" for _ in range(len(data))])
                multilang_confidence_list.extend([torch.tensor(-1.0) for _ in range(len(data))])
            
            # English prediction (fallback)
            if "english" in self.supported_languages:
                with torch.cuda.amp.autocast() and torch.inference_mode():
                    logits = self.eng_parseq(data)
                pred = logits.softmax(-1)
                eng_preds, eng_confidence = self.eng_tokenizer.decode(pred)
                eng_label_list.extend(eng_preds)
                eng_confidence_list.extend(eng_confidence)
            else:
                eng_label_list.extend(["" for _ in range(len(data))])
                eng_confidence_list.extend([torch.tensor(-1.0) for _ in range(len(data))])
        
        # Process results
        text_list = []
        conf_list = []
        
        for ml_l, ml_c, e_l, e_c in zip(multilang_label_list, multilang_confidence_list, 
                                        eng_label_list, eng_confidence_list):
            ml_conf = torch.mean(ml_c).detach().cpu().numpy().item()
            eng_conf = torch.mean(e_c).detach().cpu().numpy().item()
            
            # Choose best prediction
            if ml_conf >= eng_conf and ml_conf >= self.recognize_thres:
                # Process multi-language text (handle Tamil decoding if needed)
                if "tamil" in self.supported_languages and any(c in ml_l for c in "அஆஇஈஉஊஎஏஐஒஓஔ"):
                    ml_c_np = ml_c.detach().cpu().numpy()
                    processed_text = self.decode_file_name(ml_l, ml_c_np)
                else:
                    processed_text = ml_l
                
                text_list.append(processed_text)
                conf_list.append(ml_conf)
            elif eng_conf >= self.recognize_thres:
                text_list.append(e_l)
                conf_list.append(eng_conf)
            else:
                text_list.append("")
                conf_list.append(0.0)
        
        torch.cuda.empty_cache()
        return text_list, conf_list

# Usage example
def create_multilang_ocr(languages=["tamil", "english", "hindi"]):
    """Create multi-language OCR instance"""
    multilang_model_path = "models/multilang_parseq.pt"
    
    ocr = MultiLangOCR(
        detect=True,
        lang=languages,
        multilang_model_path=multilang_model_path,
        batch_size=16
    )
    
    return ocr

if __name__ == "__main__":
    # Example usage
    ocr = create_multilang_ocr(["tamil", "english", "hindi"])
    
    # Test with image
    # results = ocr.predict("test_image.jpg")
    # print(results)