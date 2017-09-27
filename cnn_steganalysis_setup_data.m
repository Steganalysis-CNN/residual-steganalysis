function imdb= cnn_steganalysis_setup_data()
% 该函数实现对数据读取参数的初始化
   imdb.coverDir = 'F:\BOSS\crop_BOSS\cover\';
   imdb.stegoDir = 'F:\BOSS\crop_BOSS\suniward_04\';
        
  % descriptions to the image database
  imdb.meta.sets = {'train', 'val', 'test'};
  imdb.meta.classes = {1,2};
  
  % details to the image database
  set = [ones(1,30000) 2*ones(1,10000)];
  index = randperm(length(set));
  imdb.images.set = set(index);
  save('index_suniward04.mat','index');
end