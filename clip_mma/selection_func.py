import seaborn as sns
from PIL import Image

from encoder import ImageEncoder
from handle_umap import UMAP_Module



CROP_RATE_LIST = np.arange(0.5,1.05,0.05)
ONE_FIG_SIZE = 3
MARKERSIZE  = 8
DISPLAY_IMAGE_NUMBER = 10



def crop_center(pil_img, rate):
    img_width, img_height = pil_img.size
    crop_width = int(rate*img_width)
    crop_height = int(rate*img_height)
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))


def visualize_latent_space(modelname_dict,model_feature,save_dir):
    """
    The 2D plot of embedding vectors of pattern images  are shown by UMAP
    """
    fig, ax = plt.subplots(figsize=(10,10))
    for i, model_name in modelname_dict.items():
        ax.scatter(model_feature[category_label==i,0], model_feature[category_label==i,1],
                label=model_name, s=MARKERSIZE)
        ax.autoscale()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig(os.path.join(save_dir,"umap_classified_by_models.pdf"))
    plt.show()

def visualize_similarity_in_latent_space(model_feature,target_feature,save_dir):
    """
    the plot of the target image is added on UMAP 2D plot
    and the points are colored based on their similarity scores.
    """
    fig, ax = plt.subplots(figsize=(13,10))
    m = ax.scatter(model_feature[:,0],model_feature[:,1],
                c=sim,cmap="jet", s=MARKERSIZE)
    m2 = ax.scatter(target_feature[0,0],target_feature[0,1],
                    color="black",s=250,marker="*",label="target")
    ax.autoscale()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.colorbar(m,label="similarity")
    plt.legend(handles=[m2],labels=["Target"])
    plt.savefig(os.path.join(save_dir,"umap_with_similarity.pdf"))
    plt.show()


def display_highrank_images(pil_img, image_path_list, sim, save_dir,n=10):
    """
    N number of pattern images with the highest similarity scores are shown.
    """
    highscore_idx = np.argsort(sim)[::-1]
    
    ori_img = np.array(pil_img)
    plt.imshow(ori_img)
    plt.title("target image")
    plt.show()

    raws = np.ceil(n/5)
    count = 0
    for j in range(raws):
        fig = plt.figure(figsize=(3*5,3))
        for i,idx in enumerate(highscore_idx[5*j:5*(j+1)]):
            image_path = image_path_list[idx]
            image = cv2.imread(image_path)
            ax = fig.add_subplot(1,5,i+1)
            ax.imshow(image,cmap="gray")
            ax.set_title("Rank"+str(1+5*j+i)+": "+image_path.split("/")[-2]+"\n"+"Similarity: "+str(round(sim[::-1][i+(5*j)],5)))
            ax.set_axis_off()
            count += 1
            if count == n:
                break
        plt.savefig(os.path.join(save_dir,"rank_"+str(1+j*5)+"_"+str(1+(j+1)*5)+".pdf"))
        plt.show()

def display_histogram(modelname_dict,sim,save_dir):
    """
    The histogram of the similarity scores of all the mathematical models and those of each model only are shown.
    """
    plt.hist(sim)
    plt.title("similarity histogram")
    plt.show()

    sns.set()
    for i, model_name in modelname_dict.items():
        fig, ax = plt.subplots(figsize=(3,3))

        scores = sim[category_label==i]
        plt.hist(scores[scores<1.0],
                density=True,
                alpha=0.5,
                bins = np.arange(0.7,1.01,0.02)
                )
        sns.kdeplot(scores[scores<1.0],
                    label=model_name,
                    bw_adjust=0.6
                    )
        plt.title(model_name)
        plt.xlim([0.7,1.0])
        plt.savefig(os.path.join(save_dir,model_name+"_similarity_histogram.pdf"))
        plt.show()

    plt.rcdefaults()

def measure_similarity(pil_img, config, save_dir, auto_crop=True):
    encoder = ImageEncoder(cofig)

    all_model_data = encoder.load_all_data()
    model_embeddings = all_model_data.embeddings
    image_path_list = all_model_data.image_path
    modelname_dict = all_model_data.modelname_dict
    category_label = all_model_data.category_label

    # If auto_crop is True, the proportion of center-crop is decided to maximize similarity of the most similary patten image to the target.
    if auto_crop:
        max_sim_lst = []
        for r in CROP_RATE_LIST:
            cropped_img = crop_center(pil_img,r)
            target_vector = encoder.embed_single_image(cropped_img).unsqueeze(0)
            sim = torch.nn.functional.cosine_similarity(target_vector,model_embeddings).detach().cpu().numpy()
            max_sim_lst.append(np.max(sim))

        best_rate = CROP_RATE_LIST[np.argmax(max_sim_lst)]
        pil_img = crop_center(pil_img, best_rate)

    target_vector = encoder.embed_single_image(pil_img).unsqueeze(0)
    sim = torch.nn.functional.cosine_similarity(target_vector,model_embeddings).detach().cpu().numpy()

    model_embeddings = model_embeddings.detach().cpu().numpy()
    target_vector = target_vector.detach().cpu().numpy()

    umap_handler = UMAP_Module(config)
    mapper = umap_handler.load_umap_mapper()
    model_feature = mapper.transform(model_embedings)
    target_feature = mapper.transform(target_vector)

    #The results are shown and saved in the following.
    visualize_latent_space(modelname_dict,model_feature,save_dir)
    visualize_similarity_in_latent_space(model_feature,target_feature,save_dir)
    display_highrank_images(pil_img, image_path_list, sim, save_dir,DISPLAY_IMAGE_NUMBER)
    display_histogram(modelname_dict,sim,save_dir)



def model_selection_preprocessing(config):
    encoder = ImageEncoder(config)
    encoder.embed_whole_data()
    all_model_data = encoder.load_all_data()
    umap_handler = UMAP_Module(config)
    mapper = umap_handler.fit(all_model_data)
    umap_handler.save_umap_mapper(mapper)