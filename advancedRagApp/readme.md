# في الملخص:

## نوع الـ Naive RAG (التلقائي البسيط):

- **طريقة التجزئة:**  
  يستخدم تقطيع الجمل (`SentenceSplitter`) بشكل مباشر، مما يؤدي إلى استخراج نصوص كبيرة وغير مركزة أحياناً.

- **المخرجات:**  
  ينتج عنه ملخص طويل وتفصيلي، قد يحتوي على نصوص زائدة وغير ضرورية.

- **استعمال التوكنات:**  
  يستهلك عدد أكبر من التوكنات سواءً في الإدخال أو الإخراج لأنه يمرر كتل نصية كبيرة إلى النموذج.

- **الزمن:**  
  يستغرق وقتاً أطول لمعالجة الاستعلام (حوالي 258 ثانية في المثال).

---

## نوع الـ Advanced RAG (المتقدم):

- **طريقة التجزئة:**  
  يستخدم `SentenceWindowNodeParser` الذي يجمع بين الجمل في نوافذ (windows) متداخلة، مما يساعد على استخلاص سياق أكثر تركيزاً.

- **استرجاع البيانات:**  
  يعتمد على البحث الهجين (hybrid search) الذي يوازن بين البحث القائم على المتجهات والبحث بالكلمات المفتاحية، مما ينتج عنه نصوص أكثر دقة.

- **إعادة التصنيف (Re-ranking):**  
  يستخدم مُعيد تصنيف (reranker) لتحديد أكثر الأجزاء صلة بالنص المطلوب، مما يؤدي إلى تقليل النصوص غير الضرورية.

- **المخرجات:**  
  يعطي إجابة مختصرة ومركزة تحتوي على المعلومات الأساسية فقط.

- **استعمال التوكنات:**  
  يوفر التوكنات سواءً في الإدخال أو الإخراج لأن النصوص المرسلة إلى النموذج تكون أكثر تحديداً ولا تحتوي على بيانات زائدة.

- **الزمن:**  
  يستغرق وقتاً أقل لمعالجة الاستعلام (حوالي 50 ثانية في المثال).

---

## الفرق الأساسي:

النظام المتقدم يقوم بتحسين:
1. **مرحلة ما قبل الاسترجاع:**  
   بتقسيم النصوص بطريقة أكثر فاعلية.
2. **مرحلة الاسترجاع:**  
   باستخدام البحث الهجين.
3. **مرحلة ما بعد الاسترجاع:**  
   بإعادة التصنيف.

مما يؤدي إلى:
- تقليل حجم النصوص المدخلة إلى النموذج وخروجه.
- تقليل عدد التوكنات المستخدمة.
- توفير الوقت.