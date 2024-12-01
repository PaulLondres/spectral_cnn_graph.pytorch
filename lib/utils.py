import gensim
import sklearn, sklearn.datasets
import sklearn.naive_bayes, sklearn.linear_model, sklearn.svm, sklearn.neighbors, sklearn.ensemble
import matplotlib.pyplot as plt
import scipy.sparse
import numpy as np
import time, re


# Helpers to process text documents.


class TextDataset(object):
    def clean_text(self, num='substitute'):
        # TODO: stemming, lemmatisation
        for i,doc in enumerate(self.documents):
            # Digits.
            if num == 'spell':
                doc = doc.replace('0', ' zero ')
                doc = doc.replace('1', ' one ')
                doc = doc.replace('2', ' two ')
                doc = doc.replace('3', ' three ')
                doc = doc.replace('4', ' four ')
                doc = doc.replace('5', ' five ')
                doc = doc.replace('6', ' six ')
                doc = doc.replace('7', ' seven ')
                doc = doc.replace('8', ' eight ')
                doc = doc.replace('9', ' nine ')
            elif num == 'substitute':
                # All numbers are equal. Useful for embedding (countable words) ?
                doc = re.sub('(\\d+)', ' NUM ', doc)
            elif num == 'remove':
                # Numbers are uninformative (they are all over the place). Useful for bag-of-words ?
                # But maybe some kind of documents contain more numbers, e.g. finance.
                # Some documents are indeed full of numbers. At least in 20NEWS.
                doc = re.sub('[0-9]', ' ', doc)
            # Remove everything except a-z characters and single space.
            doc = doc.replace('$', ' dollar ')
            doc = doc.lower()
            doc = re.sub('[^a-z]', ' ', doc)
            doc = ' '.join(doc.split())  # same as doc = re.sub('\s{2,}', ' ', doc)
            self.documents[i] = doc

    def vectorize(self, **params):
        # TODO: count or tf-idf. Or in normalize ?
        vectorizer = sklearn.feature_extraction.text.CountVectorizer(**params)
        self.data = vectorizer.fit_transform(self.documents)
        self.vocab = vectorizer.get_feature_names_out()
        assert len(self.vocab) == self.data.shape[1]
    
    def data_info(self, show_classes=False):
        N, M = self.data.shape
        sparsity = self.data.nnz / N / M * 100
        print('N = {} documents, M = {} words, sparsity={:.4f}%'.format(N, M, sparsity))
        if show_classes:
            for i in range(len(self.class_names)):
                num = sum(self.labels == i)
                print('  {:5d} documents in class {:2d} ({})'.format(num, i, self.class_names[i]))
        
    def show_document(self, i):
        label = self.labels[i]
        name = self.class_names[label]
        try:
            text = self.documents[i]
            wc = len(text.split())
        except AttributeError:
            text = None
            wc = 'N/A'
        print('document {}: label {} --> {}, {} words'.format(i, label, name, wc))
        try:
            vector = self.data[i,:]
            for j in range(vector.shape[1]):
                if vector[0,j] != 0:
                    print('  {:.2f} "{}" ({})'.format(vector[0,j], self.vocab[j], j))
        except AttributeError:
            pass
        return text
    
    def keep_documents(self, idx):
        """Keep the documents given by the index, discard the others."""
        self.documents = [self.documents[i] for i in idx]
        self.labels = self.labels[idx]
        self.data = self.data[idx,:]

    def keep_words(self, idx):
        """Keep the documents given by the index, discard the others."""
        self.data = self.data[:,idx]
        self.vocab = [self.vocab[i] for i in idx]
        try:
            self.embeddings = self.embeddings[idx,:]
        except AttributeError:
            pass

    def remove_short_documents(self, nwords, vocab='selected'):
        """Remove a document if it contains less than nwords."""
        if vocab == 'selected':
            # Word count with selected vocabulary.
            wc = self.data.sum(axis=1)
            wc = np.squeeze(np.asarray(wc))
        elif vocab == 'full':
            # Word count with full vocabulary.
            wc = np.empty(len(self.documents), dtype=int)
            for i,doc in enumerate(self.documents):
                wc[i] = len(doc.split())
        idx = np.argwhere(wc >= nwords).squeeze()
        self.keep_documents(idx)
        return wc
        
    def keep_top_words(self, M, Mprint=20):
        """Keep in the vocaluary the M words who appear most often."""
        freq = self.data.sum(axis=0)
        freq = np.squeeze(np.asarray(freq))
        idx = np.argsort(freq)[::-1]
        idx = idx[:M]
        self.keep_words(idx)
        print('most frequent words')
        for i in range(Mprint):
            print('  {:3d}: {:10s} {:6d} counts'.format(i, self.vocab[i], freq[idx][i]))
        return freq[idx]
    
    def normalize(self, norm='l1'):
        """Normalize data to unit length."""
        # TODO: TF-IDF.
        data = self.data.astype(np.float64)
        self.data = sklearn.preprocessing.normalize(data, axis=1, norm=norm)
        
    def embed(self, filename=None, size=100):
        """Embed the vocabulary using pre-trained vectors."""
        if filename:
            model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)
            size = model.vector_size
        else:
            class Sentences(object):
                def __init__(self, documents):
                    self.documents = documents
                def __iter__(self):
                    for document in self.documents:
                        yield document.split()
            model = gensim.models.Word2Vec(sentences=Sentences(self.documents), vector_size=size)
        self.embeddings = np.empty((len(self.vocab), size))
        keep = []
        not_found = 0
        for i,word in enumerate(self.vocab):
            try:
                self.embeddings[i,:] = model.wv.key_to_index[word]
                keep.append(i)
            except KeyError:
                not_found += 1
        print('{} words not found in corpus'.format(not_found, i))
        self.keep_words(keep)

class Text20News(TextDataset):
    def __init__(self, **params):
        dataset = sklearn.datasets.fetch_20newsgroups(**params)
        self.documents = dataset.data
        self.labels = dataset.target
        self.class_names = dataset.target_names
        assert max(self.labels) + 1 == len(self.class_names)
        N, C = len(self.documents), len(self.class_names)
        print('N = {} documents, C = {} classes'.format(N, C))

class TextRCV1(TextDataset):
    def __init__(self, **params):
        dataset = sklearn.datasets.fetch_rcv1(**params)
        self.data = dataset.data
        self.target = dataset.target
        self.class_names = dataset.target_names
        assert len(self.class_names) == 103  # 103 categories according to LYRL2004
        N, C = self.target.shape
        assert C == len(self.class_names)
        print('N = {} documents, C = {} classes'.format(N, C))

    def remove_classes(self, keep):
        ## Construct a lookup table for labels.
        labels_row = []
        labels_col = []
        class_lookup = {}
        for i,name in enumerate(self.class_names):
            class_lookup[name] = i
        self.class_names = keep

        # Index of classes to keep.
        idx_keep = np.empty(len(keep))
        for i,cat in enumerate(keep):
            idx_keep[i] = class_lookup[cat]
        self.target = self.target[:,idx_keep]
        assert self.target.shape[1] == len(keep)

    def show_doc_per_class(self, print_=False):
        """Number of documents per class."""
        docs_per_class = np.array(self.target.astype(np.uint64).sum(axis=0)).squeeze()
        print('categories ({} assignments in total)'.format(docs_per_class.sum()))
        if print_:
            for i,cat in enumerate(self.class_names):
                print('  {:5s}: {:6d} documents'.format(cat, docs_per_class[i]))
        plt.figure(figsize=(17,5))
        plt.plot(sorted(docs_per_class[::-1]),'.')

    def show_classes_per_doc(self):
        """Number of classes per document."""
        classes_per_doc = np.array(self.target.sum(axis=1)).squeeze()
        plt.figure(figsize=(17,5))
        plt.plot(sorted(classes_per_doc[::-1]),'.')

    def select_documents(self):
        classes_per_doc = np.array(self.target.sum(axis=1)).squeeze()
        self.target = self.target[classes_per_doc==1]
        self.data = self.data[classes_per_doc==1, :]

        # Convert labels from indicator form to single value.
        N, C = self.target.shape
        target = self.target.tocoo()
        self.labels = target.col
        assert self.labels.min() == 0
        assert self.labels.max() == C - 1

        # Bruna and Dropout used 2 * 201369 = 402738 documents. Probably the difference btw v1 and v2.
        #return classes_per_doc
