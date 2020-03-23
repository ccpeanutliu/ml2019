# ml2019
李宏毅教授的機器學習課程作業

## hw1
(linear regression)
- 本次作業的資料是從中央氣象局網站下載的真實觀測資料，利用linear regression或其他方法預測PM2.5的數值。
- 作業使用豐原站的觀測記錄，分成train set跟test set，train set是豐原站每個月的前20天所有資料。test set則是從豐原站剩下的資料中取樣出來。
- train set：每個月前20天的完整資料。
- test set：從剩下的10天資料中取樣出連續的10小時為一筆，前九小時的所有觀測數據當作feature，第十小時的PM2.5當作answer。一共取出240筆不重複的test data，請根據feauure預測這240筆的PM2.5。

## hw2
(logistic regression)
- 本次作業是需要從給定的個人資訊，預測此人的年收入是否大於50K。
- 共有32561筆訓練資料，16281筆測試資料。

## hw3
(Image Sentiment Classification)
- training dataset為兩萬八千張左右48x48 pixel的圖片，以及每一張圖片的表情label（注意：每張圖片都會唯一屬於一種表情）。
- 總共有七種可能的表情（0：生氣, 1：厭惡, 2：恐懼, 3：高興, 4：難過, 5：驚訝, 6：中立(難以區分為前六種的表情))。

## hw4
(model description)
- 使用hw3 model進行預測，並視覺化model、做出saliency map。
- 再使用Lime來理解一張圖片，圖片的哪些部分提供model較多的資訊。

## hw5
(model attack)
- training set是一堆圖片，想辦法對圖片進行一些攻擊，讓助教那邊的model判斷錯誤。
- 圖片更改幅度必須小於某一個特定值。

## hw6
(RNN、BOW、word2vec) 
- Dcard 提供的匿名留言資料。
- 實作 Recurrent Neural Network (RNN) 與 Bag of Word (BOW) 來判斷留言是否為惡意留言（人身攻擊, 仇恨言論, etc.）。

## hw7
(unsupervised learning)
- 實作unsupervised learning, 判斷兩張圖片是否來自同一個dataset
- 圖片共有40000張
- 測資共1000000筆

## hw8
(model compression、small model prediction)
- 實作model compression，要將Image Sentiment Classification的model做壓縮，使其不超過一定大小的情況下，accuracy的下降一定範圍內
- 圖片共有約35000張，其中約7000張用於testing
