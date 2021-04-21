import React from 'react';
import { fireEvent, getByTestId, getByText, render,screen, waitFor } from '@testing-library/react';


import Enzyme from 'enzyme';
//import Adapter from 'enzyme-adapter-react-16';
import Adapter from '@wojtekmaj/enzyme-adapter-react-17'
Enzyme.configure({ adapter: new Adapter() });

import {shallow} from 'enzyme'
import {UploadDatasetForm} from '../components/forms/dataset.js'

import userEvent from "@testing-library/user-event";

/**
 * Test What user would see
 */
 test('renders the form correctly', () => {
    render(<UploadDatasetForm />);
    const nameLabel = screen.getByText('Dataset Name')
    const descriptionLabel = screen.getByText('Description');
    const datasetLabel = screen.getByText('Dataset');
    const typeLabel = screen.getByText('Data Type');
    
    expect(nameLabel).toBeInTheDocument();
    expect(descriptionLabel).toBeInTheDocument();
    expect(datasetLabel).toBeInTheDocument();
    expect(typeLabel).toBeInTheDocument();
    
    const registerBtn = screen.getByTestId("submit")
    //const chooseFile = screen.getByTestId("file")
    expect(registerBtn).toBeInTheDocument();
    //expect(chooseFile).toBeInTheDocument();
    
    //execute click
    userEvent.click(registerBtn);
   // userEvent.click(chooseFile);
  })




const simulateChangeOnInput = (wrapper,inputname, inputSelector, newValue) => {
    let input = wrapper.find(inputSelector)
    input.simulate('change',{
        target: {name:inputname,value: newValue},
    })
    return wrapper.find(inputSelector)
}



describe('uploading a file', () => {
    

    
    let file;

    beforeEach(() => {
        file = new File(['(⌐□_□)'], 'chucknorris.py', { type: 'script/py' });
    });
    
    test('file upload',async () =>{

        const { getByTestId } = render(<UploadDatasetForm />);
        
        // get the upload button
        let uploader = getByTestId('file');

        // simulate ulpoad event and wait until finish 
        await waitFor(() =>
        fireEvent.change(uploader, {
            target: { files: [file] },
        })
        );
        // get the same uploader from the dom    
        let uploadedFile = document.getElementById('fileUpload');
        // check if the file is there
        expect(uploadedFile.files[0].name).toBe('chucknorris.py');
        expect(uploadedFile.files.length).toBe(1);
        
        
        
           
    });
    
    

    it('lets me fill out the form', () => {
        const wrapper = shallow (<UploadDatasetForm/>)
        const nameInput = simulateChangeOnInput(wrapper,'name','input#name-input','ICU data')  //second argument is the id of the input field 
        const descriptionInput = simulateChangeOnInput(wrapper,'description','input#description-input','Simple dataset')
        const dataTypeInput = simulateChangeOnInput(wrapper,'dataType','input#dataType-input','abcd')
        
        expect(nameInput.props().value).toEqual('ICU data')
        expect(descriptionInput.props().value).toEqual('Simple dataset')
        expect(dataTypeInput.props().value).toEqual('abcd')

        

    });
    

    
})
