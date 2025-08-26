// namespace GcRT{
// namespace memory{
//     /**
//      * TCache是线程本地缓存，线程应用申请中、小型对象时，首先从TCache中申请。
//      * 当线程应用析构某些内存时，如果是中小型对象，会先将内存交到TCache这里，然后由TCache缓存。
//      * TCache内部类似于一个buddy
//      */
//     class TCache{
//     public:
//     };


// }

// }