from keras.initializers import *
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.callbacks import *
from keras.regularizers import *


PROTEIN_SEQUENCE_INFO_CHANNELS = 10
LIGAND_SEQUENCE_INFO_CHANNELS = 10

def conv_pool_block(x, num_filter):
	x = Conv2D(num_filter, (1, 3),
		strides=(1, 1),
		use_bias=False,
		kernel_regularizer=l2(l=0.1),
		kernel_initializer=glorot_uniform(),
		data_format='channels_last',
		padding='same')(x)
	x = BatchNormalization()(x)
	x = LeakyReLU()(x)

	x = Conv2D(num_filter, (1, 3),
		strides=(1, 1),
		use_bias=False,
		kernel_regularizer=l2(l=0.1),
		kernel_initializer=glorot_uniform(),
		padding='same')(x)
	x = BatchNormalization()(x)
	x = LeakyReLU()(x)

	x = MaxPooling2D(pool_size=(1, 2))(x)
	x = Dropout(0.5)(x)

	return x

def protein_net(protein_input):
	out = conv_pool_block(protein_input, 64)
	out = conv_pool_block(out, 128)
	out = conv_pool_block(out, 256)
	out = conv_pool_block(out, 512)
	out = conv_pool_block(out, 1024)

	out = Conv2D(PROTEIN_SEQUENCE_INFO_CHANNELS, (1, 3),
			strides=(1, 1),
			use_bias=False,
			kernel_regularizer=l2(l=0.1),
			kernel_initializer=glorot_uniform(),
			padding='valid')(out)
	out = BatchNormalization()(out)
	out = LeakyReLU()(out)

	return out

def ligand_net(x):
	x = Conv2D(64, (1, 3),
			strides=(1, 1),
			use_bias=False,
			kernel_regularizer=l2(l=0.1),
			kernel_initializer=glorot_uniform(),
			padding='same')(x)
	x = BatchNormalization()(x)
	x = LeakyReLU()(x)

	x = Conv2D(LIGAND_SEQUENCE_INFO_CHANNELS, (1, 1),
			strides=(1, 1),
			use_bias=False,
			kernel_regularizer=l2(l=0.1),
			kernel_initializer=glorot_uniform(),
			padding='valid')(x)
	x = BatchNormalization()(x)
	x = LeakyReLU()(x)

	return x

def context_extract_net(x):
	# padding
	x = ZeroPadding2D((0, 2), data_format='channels_last')(x)
	
	# conv 1
	x = LocallyConnected2D(512, (1, 2),
		strides=(1, 1),
		use_bias=False,
		kernel_regularizer=l2(l=0.1),
		kernel_initializer=glorot_uniform(),
		padding='valid')(x)
	x = BatchNormalization()(x)
	x = LeakyReLU()(x)

	# pool
	x = MaxPooling2D(pool_size=(1, 2))(x)
	x = Dropout(0.5)(x)

	# conv 2
	x = LocallyConnected2D(512, (1, 2),
		strides=(1, 1),
		use_bias=False,
		kernel_regularizer=l2(l=0.1),
		kernel_initializer=glorot_uniform(),
		padding='valid')(x)
	x = BatchNormalization()(x)
	x = LeakyReLU()(x)

	x = Flatten()(x)

	return x


def regression_net(x):
	# fully
	x = Dropout(0.5)(x)
	x = Dense(100,
		name="regression_dense",
		kernel_regularizer=l2(l=0.1),
		use_bias=True,
		kernel_initializer=glorot_uniform())(x)
	x = LeakyReLU()(x)
	x = Dropout(0.5)(x)

	# output
	output = Dense(1,
		name="regression_output",
		kernel_regularizer=l2(l=0.1),
		use_bias=False,
		kernel_initializer=glorot_uniform())(x)

	return output


def classifiy_net(x):
	# fully
	x = Dropout(0.5)(x)
	x = Dense(100,
		name="classify_dense",
		kernel_regularizer=l2(l=0.1),
		use_bias=True,
		kernel_initializer=glorot_uniform())(x)
	x = LeakyReLU()(x)
	x = Dropout(0.5)(x)

	# 
	output = Dense(1,
		name="classify_output",
		kernel_regularizer=l2(l=0.1),
		use_bias=False,
		activation="sigmoid",
		kernel_initializer=glorot_uniform())(x)

	return output

def model_config():
	# protein input
	protein_input = Input(shape=(1, 372, 21), name='protein')
	protein_feature = protein_net(protein_input)

	# ligand input
	ligand_input = Input(shape=(1, 9, 20), name='ligand')
	ligand_feature = ligand_net(ligand_input)

	# binding context
	merge_input = concatenate([ligand_feature, protein_feature])
	context = context_extract_net(merge_input)


	# regression predictior
	regression = regression_net(context)

	# classification
	classify = classifiy_net(context)

	model = Model(inputs=[protein_input, ligand_input], outputs=[regression, classify])

	return model


def protein_model():
	protein_input = Input(shape=(1, 372, 21), name='protein')
	protein_feature = protein_net(protein_input)
	model = Model(inputs=[protein_input], outputs=[protein_feature])

	return model

def ligand_model():
	ligand_input = Input(shape=(1, 9, 20), name='ligand')
	ligand_feature = ligand_net(ligand_input)
	model = Model(inputs=[ligand_input], outputs=[ligand_feature])

	return model

if __name__ == '__main__':
	from keras.utils import plot_model
	m = model_config()
	m.summary()
	plot_model(m, to_file='model.png')
	pass