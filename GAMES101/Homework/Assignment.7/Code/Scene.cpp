/*
 * @Author: Zero
 * @Date: 2021-02-02 15:30:16
 * @LastEditTime: 2021-02-02 18:01:50
 * @Description: 
 * @可以输入预定的版权声明、个性签名、空行等
 */
//
// Created by Göksu Güvendiren on 2019-05-14.
//

#include "Scene.hpp"


void Scene::buildBVH() {
    printf(" - Generating BVH...\n\n");
    this->bvh = new BVHAccel(objects, 1, BVHAccel::SplitMethod::NAIVE);
}

Intersection Scene::intersect(const Ray &ray) const
{
    return this->bvh->Intersect(ray);
}

void Scene::sampleLight(Intersection &pos, float &pdf) const
{
    float emit_area_sum = 0;
    for (uint32_t k = 0; k < objects.size(); ++k) {
        if (objects[k]->hasEmit()){
            emit_area_sum += objects[k]->getArea();
        }
    }
    float p = get_random_float() * emit_area_sum;
    emit_area_sum = 0;
    for (uint32_t k = 0; k < objects.size(); ++k) {
        if (objects[k]->hasEmit()){
            emit_area_sum += objects[k]->getArea();
            if (p <= emit_area_sum){
                objects[k]->Sample(pos, pdf);
                break;
            }
        }
    }
}

bool Scene::trace(
        const Ray &ray,
        const std::vector<Object*> &objects,
        float &tNear, uint32_t &index, Object **hitObject)
{
    *hitObject = nullptr;
    for (uint32_t k = 0; k < objects.size(); ++k) {
        float tNearK = kInfinity;
        uint32_t indexK;
        Vector2f uvK;
        if (objects[k]->intersect(ray, tNearK, indexK) && tNearK < tNear) {
            *hitObject = objects[k];
            tNear = tNearK;
            index = indexK;
        }
    }


    return (*hitObject != nullptr);
}

// Implementation of Path Tracing
Vector3f Scene::castRay(const Ray &ray, int depth) const
{
    // TO DO Implement Path Tracing Algorithm here
	
    Vector3f ldir = {0, 0, 0};			// 对光源直接采样得到的光线
    Vector3f lindir = {0, 0, 0};		// 其他物体反射过来的光线
	
    // 得到光线和场景中物体的交点
    Intersection objectInter = intersect(ray);
    // 没有交点，说明没有物体，直接返回即可
    if(!objectInter.happened) return {};

	// 物体自发光，返回其发射出的光线
    if(objectInter.m->hasEmission()) return objectInter.m->getEmission();
	
    // 对交点位置采样，lightpdf只是初始化，会在Sphere.hpp的Sample函数中计算
    Intersection lightInter;
    float lightPdf = 0.0f;
    sampleLight(lightInter, lightPdf);

	// 物体到光源的向量
    Vector3f obj2light = lightInter.coords - objectInter.coords;
    Vector3f obj2lightdir = obj2light.normalized();

    // |x-p|^2
    float distancePow2 = obj2light.x * obj2light.x + obj2light.y * obj2light.y + obj2light.z * obj2light.z;
	// Shoot a ray from p to x；交点位置，指向光线方向
    Ray obj2lightray = {objectInter.coords, obj2lightdir};
    
    // 光线和场景的交点
    Intersection t = intersect(obj2lightray);
    // 光线中间没有碰到其他物体，即两向量长度相同
    if(t.distance - obj2light.norm() > -EPSILON) {
        // L_i * f_r * cos_theta * cos_theta_x / |x-p|^2 / pdf_light
        // emit * eval(wo,ws,N) * dot(ws,N) * dot(ws,NN) / |x-p|^2 / pdf_light
        // eval即物体的BRDF
        ldir = lightInter.emit * objectInter.m->eval(ray.direction, obj2lightdir, objectInter.normal) * dotProduct(obj2lightdir, objectInter.normal) * dotProduct(-obj2lightdir, lightInter.normal) / distancePow2 / lightPdf;
    }
	
    // 产生随机数，大于概率就直接返回，不再弹射光线
    if(get_random_float() > RussianRoulette) return ldir;

	// 从一个物体反射到另一个物体，在像素中直接采样多次，N=1，而不是一个点多次采样光线
    Vector3f obj2nextobjdir = objectInter.m->sample(ray.direction, objectInter.normal).normalized();
    Ray obj2nextobjray = {objectInter.coords, obj2nextobjdir};

    // 反射光线和物体求交
    Intersection nextObjInter = intersect(obj2nextobjray);
    // 相交且无自发光
    if(nextObjInter.happened && !nextObjInter.m->hasEmission()) {

        // 生成PDF
        float pdf = objectInter.m->pdf(ray.direction, obj2nextobjdir, objectInter.normal);
        // 递归求弹射光线
        lindir = castRay(obj2nextobjray, depth + 1) * objectInter.m->eval(ray.direction, obj2nextobjdir, objectInter.normal) * dotProduct(obj2nextobjdir, objectInter.normal) / pdf / RussianRoulette;
    }

    return ldir + lindir;
}