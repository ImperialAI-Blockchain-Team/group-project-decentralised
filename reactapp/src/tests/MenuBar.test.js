import React from 'react';
import { fireEvent, getByTestId, getByText, render,screen, waitFor } from '@testing-library/react';

import Enzyme from 'enzyme';
//import Adapter from 'enzyme-adapter-react-16';
import Adapter from '@wojtekmaj/enzyme-adapter-react-17'
 
Enzyme.configure({ adapter: new Adapter() });

import {shallow} from 'enzyme'
import {MenuBar} from '../components/MenuBar/MenuBar.js'





describe('MenuBar', () => {
    it('links work', () => {
        const wrapper = shallow (<MenuBar/>)

        /*
        wrapper.find("#dropdown").at(0).simulate('change',{
            target:{value:"item1",name:"item1"}
        })*/

        expect(wrapper.find('Link').at(0).prop("to")).toEqual("/about")
        expect(wrapper.find('Link').at(1).prop("to")).toEqual("/sign_up")
        expect(wrapper.find('Link').at(2).prop("to")).toEqual("/my_account")
        expect(wrapper.find('Link').at(3).prop("to")).toEqual("/register_model")
        expect(wrapper.find('Link').at(4).prop("to")).toEqual("/register_dataset")
        expect(wrapper.find('Link').at(5).prop("to")).toEqual("/browse_models")
        expect(wrapper.find('Link').at(6).prop("to")).toEqual("/browse_datasets")
        expect(wrapper.find('Link').at(7).prop("to")).toEqual("/browse_jobs")
        expect(wrapper.find('Link').at(8).prop("to")).toEqual("/test_components")
        
        
        

        

    })

    

    
})