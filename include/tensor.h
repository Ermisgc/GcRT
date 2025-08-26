#pragma once
#include <tuple>
#include <vector>
#include <array>
#include <NvInfer.h>
#include <exception>
#include <stdexcept>
#include <memory>
#include <iostream>
#include <cuda_fp16.h>
#include <type_traits>
#include <map>
#include <optional>

namespace GcRT{
    /**
     * @brief The helper function to check if a type is a vector.
     * This is a generic structure implementation.
     */
    template<typename T>
    struct is_vector: std::false_type{};

    /**
     * @brief The helper function to check if a type is a vector.
     * This is a speical structure implementation.
     */
    template<typename T>
    struct is_vector<std::vector<T>>: std::true_type{};


    // 特化定义，将数据类型与nvinfer1::DataType映射起来
    template<typename Tp>
    struct type_traits;

#define MAP_TYPE_TRAITS(type, nv_type) \
template<> struct type_traits<type> { \
    static constexpr nvinfer1::DataType dtype = nvinfer1::DataType::nv_type; \
}
    MAP_TYPE_TRAITS(float, kFLOAT);
    MAP_TYPE_TRAITS(int64_t, kINT64);
    MAP_TYPE_TRAITS(int32_t, kINT32);
    MAP_TYPE_TRAITS(bool, kBOOL);
    MAP_TYPE_TRAITS(uint8_t, kUINT8);
    MAP_TYPE_TRAITS(int8_t, kINT8);
    MAP_TYPE_TRAITS(__half, kHALF);
#undef MAP_TYPE_TRAITS

    template<typename T>
    class ConcreteTensor;

    class Tensor{
    protected:
        using DataPtr = std::shared_ptr<void>;
        DataPtr _data; //data store in host
        nvinfer1::DataType _dType;
        nvinfer1::Dims _shape;
        size_t _size;
        static inline const std::map<nvinfer1::DataType, size_t> dtype_size_map{
            {nvinfer1::DataType::kFLOAT, sizeof(float)},
            {nvinfer1::DataType::kHALF, sizeof(half)},
            {nvinfer1::DataType::kINT8, sizeof(int8_t)},
            {nvinfer1::DataType::kINT32, sizeof(int32_t)},
            {nvinfer1::DataType::kINT64, sizeof(int64_t)},
            {nvinfer1::DataType::kBOOL, sizeof(bool)},
            {nvinfer1::DataType::kUINT8, sizeof(uint8_t)}
        };
        std::string _name;

        Tensor() = default;

        template<typename T>
        static void * allocate(size_t count){
            return new T[count];
        }

        template<typename T>
        static void deallocate(void * ptr) noexcept{
            //static_cast is safer than reinterpret_cast
            delete[] static_cast<T*>(ptr);
        }

        static void * cudallocate(size_t count){
            void * ret;
            cudaError_t err = cudaMalloc(&ret, count);
            if(err == cudaSuccess) return ret;
            else return nullptr;
        }

        static void cudadeallocate(void * ptr){
            cudaFree(ptr);
        }
    public:
        virtual ~Tensor() = default;

        /**
         * @brief factory method to create an empty tensor with a given dtype and dims
         * @example auto float_tensor = GcRT::create_tensor<float>({4, {1, 3, 224, 224}});
         */
        template<typename T>
        static std::unique_ptr<Tensor> create(nvinfer1::Dims dims){
            size_t size = 1;
            for(int i = 0;i < dims.nbDims; ++i) size *= dims.d[i];

            auto tensor = std::make_unique<ConcreteTensor<T>>();
            tensor->_dType = type_traits<T>::dtype;
            tensor->_shape = dims;
            tensor->_size = size;
            tensor->_data = DataPtr(
                allocate<T>(size),
                &deallocate<T>
            );
            return tensor;
        }

        inline size_t size() const{
            return this->_size;
        } 

        inline nvinfer1::DataType type() const{
            return this->_dType;
        }

        inline size_t dim(int dim = 0) const {
            return this->_shape.d[dim];
        }

        /**
         * @brief get the dim of -2, W in N-C-W-H
         */
        inline size_t dim_w() const{
            return this->_shape.d[2];
        }

        /**
         * @brief get the dim of -1, H in N-C-W-H
         */
        inline size_t dim_h() const{
            return this->_shape.d[3];
        }


        inline bool isNull() const{
            return this->_size == 0;
        }

        inline nvinfer1::Dims dims() {
            return this->_shape;
        }

        inline DataPtr data() const {
            return _data;
        }

        inline size_t bytes() const{
            return this->_size * dtype_size_map.at(this->_dType);
        }

        inline void setName(const std::string & n){
            _name = n;
        }

        inline std::string name() const {
            return _name;
        }

        inline cudaError_t copiedFromHost(Tensor & tensor){
            return cudaMemcpy(this->_data.get(), tensor.data().get(), this->bytes() ,cudaMemcpyHostToDevice);
        }

        inline cudaError_t cupyToHost(Tensor & tensor){
            return cudaMemcpy(tensor.data().get(), this->_data.get(), this->bytes() ,cudaMemcpyDeviceToHost);
        }
    };

    template<typename _Tp>
    class ConcreteTensor final: public Tensor{
        static_assert(!std::is_same_v<_Tp, void>, "Type of Tensor cannot be void");
    
    public:
        ConcreteTensor() = default;
        ~ConcreteTensor(){}
        
        /*<--- constructors --->*/
        static std::unique_ptr<Tensor> from_pointer(_Tp * dat, nvinfer1::Dims dims){
            if(!dat) {
                return nullptr;
            }
            size_t size = 1;
            for(int i = 0;i < dims.nbDims; ++i) size *= dims.d[i];

            auto tensor = std::make_unique<ConcreteTensor>();
            tensor->_dType = type_traits<_Tp>::dtype;
            tensor->_shape = dims;
            tensor->_size = size;
            tensor->_data = DataPtr(
                dat,
                &deallocate<_Tp>
            );
            return std::move(tensor);
        }

        static std::unique_ptr<Tensor> from_vector(std::vector<_Tp> && data, nvinfer1::Dims dims){
            // recurseForVectorShape(data, 0);
            size_t size = 1;
            for(int i = 0;i < dims.nbDims; ++i) size *= dims.d[i];
            if(size != data.size()) throw std::invalid_argument("Tensor's dims and data's size do not match.");

            _Tp *raw_data = new _Tp[size];
            std::move(data.begin(), data.end(), raw_data);
            return from_pointer(raw_data, dims);
            //  tensor;
        }

    private:
        template<typename T>
        void recurseForVectorShape(const T & t, size_t index){
            if constexpr (is_vector<T>::value){  // T is a std::vector type;
                if(t.size() == 0) throw std::exception("Empty vector is not permitted for constructing Tensor");
                _shape.d[index] = t.size();
                recurseForVectorShape(t[0], index + 1);
            }
        }
    };    

    class CudaTensor final: public Tensor{
    public:
        CudaTensor() = default;
        ~CudaTensor() = default;

        static std::unique_ptr<Tensor> from_dims(nvinfer1::DataType type, nvinfer1::Dims dims){
            size_t size = 1;
            for(int i = 0;i < dims.nbDims; ++i){
                size *= dims.d[i];
            }

            auto tensor = std::unique_ptr<CudaTensor>(new CudaTensor());
            tensor->_shape = dims;
            tensor->_dType = type;
            tensor->_size = size;
            tensor->_data = DataPtr(
                cudallocate(tensor->bytes()),
                &cudadeallocate
            );
            return tensor;
        }
    };

    using TensorUniquePtr = std::unique_ptr<Tensor>;
    using TensorSharedPtr = std::shared_ptr<Tensor>;

    template<typename T>
    TensorUniquePtr createTensor(nvinfer1::Dims dims){
        return Tensor::create<T>(dims);
    }

    template<typename T>
    TensorUniquePtr createTensorFromPointer(T * data, nvinfer1::Dims dims){
        return ConcreteTensor<T>::from_pointer(data, dims);
    }

    template<typename T>
    TensorUniquePtr createTensorFromVector(std::vector<T> && data, nvinfer1::Dims dims){
        return ConcreteTensor<T>::from_vector(std::move(data), dims);
    }

    TensorUniquePtr createTensorByType(nvinfer1::DataType type, nvinfer1::Dims dims);

    TensorUniquePtr createCudaTensor(nvinfer1::DataType type, nvinfer1::Dims dims);

    template<typename T>
    struct isTensorUniquePtr_Impl : std::false_type{};

    template<typename T>
    struct isTensorUniquePtr_Impl<std::unique_ptr<T>> : std::is_base_of<Tensor, T>{};

    template<typename T>
    constexpr bool isTensorUniquePtr_v = isTensorUniquePtr_Impl<T>::value;
}  // namespace GcRT
